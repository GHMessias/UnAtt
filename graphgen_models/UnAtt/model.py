# -*- coding: utf-8 -*-
"""
(Un)balance (Att)ributed Graph Generator via Node degree Distribution — UnAtt

Principais pontos:
- Quando mimetizando um grafo (data_to_mimic != None), a contagem de arestas é
  feita em G NÃO-DIRECIONADO CANÔNICO (cada par {u,v} conta 1 vez).
- A geração adiciona arestas nas duas direções (padrão PyG), mas evita duplicatas
  com conjuntos de pares e utilitários.
- As PDFs por classe (p(nó|classe)) são obtidas pelos graus do subgrafo da classe
  em representação undirected simétrica (to_undirected), e normalizadas.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.utils import (
    degree,
    to_undirected,
    coalesce,
    remove_self_loops,
)

# Suas utils (já existentes no projeto)
from graphgen_models.UnAtt.func import (
    split_edge_index_by_label,
    count_heterogeneity,
    remap_edge_index,
    simple_undirected,
    unique_undirected_edges,
    undirected_edge_count,
    interclass_undirected_edge_count
)

def simple_undirected(edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
    """
    Converte para representação UNDIRECTED *simétrica* (contém (u,v) e (v,u)),
    remove self-loops e duplicatas exatas. Útil para calcular graus com `degree`.
    """
    if edge_index.numel() == 0:
        return edge_index

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    ei = to_undirected(edge_index, num_nodes=num_nodes)  # insere (v,u) quando necessário
    ei, _ = remove_self_loops(ei)
    ei, _ = coalesce(ei, None, num_nodes=num_nodes)      # ordena e remove duplicatas exatas
    return ei


def unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Converte para representação UNDIRECTED *canônica* com 1 coluna por aresta {u,v}:
    - remove self-loops
    - ordena extremos por coluna: (u,v) -> (min(u,v), max(u,v))
    - remove duplicatas (coalesce)
    Retorno: edge_index com cada par {u,v} uma única vez.
    """
    if edge_index.numel() == 0:
        return edge_index

    u, v = edge_index[0], edge_index[1]
    mask = (u != v)                           # remove self-loop
    u, v = u[mask], v[mask]

    u2 = torch.minimum(u, v)                  # (min, max) por coluna
    v2 = torch.maximum(u, v)
    ei = torch.stack([u2, v2], dim=0)

    num_nodes = int(max(int(ei.max().item()) + 1, 1))
    ei, _ = coalesce(ei, None, num_nodes=num_nodes)  # remove duplicatas de colunas idênticas
    return ei


def undirected_edge_count(edge_index: torch.Tensor) -> int:
    """Conta arestas NÃO-DIRECIONADAS (cada {u,v} uma vez)."""
    return int(unique_undirected_edges(edge_index).size(1))


def interclass_undirected_edge_count(edge_index: torch.Tensor, y: torch.Tensor) -> int:
    """Conta arestas NÃO-DIRECIONADAS {u,v} com y[u] != y[v]."""
    ei = unique_undirected_edges(edge_index)
    u, v = ei[0], ei[1]
    return int((y[u] != y[v]).sum().item())


# =============================================================================
# Modelo UnAtt
# =============================================================================

class unatt:
    def __init__(
        self,
        number_of_nodes: list[int] | int = 0,             # |V_c| por classe (se não mimetizar)
        k: int | None = None,                             # nº de classes (se não mimetizar)
        edge_distributions: list[str] | None = None,      # famílias ('uniform','normal','weibull',...)
        noise_class_distribution: torch.Tensor | None = None,  # P(classe) para interclasse
        data_to_mimic: Data | None = None,                # grafo PyG para mimetizar
    ):
        self.data_to_mimic = data_to_mimic
        self.number_of_nodes = torch.tensor(number_of_nodes) if isinstance(number_of_nodes, (list, tuple, np.ndarray)) else number_of_nodes

        if data_to_mimic is None:
            # ---------------- MODO "DO ZERO" ----------------
            if self.number_of_nodes is None or (isinstance(self.number_of_nodes, torch.Tensor) and self.number_of_nodes.numel() == 0):
                raise ValueError("Defina number_of_nodes quando data_to_mimic=None.")

            self.k = int(k)
            self.edge_distributions = edge_distributions or ["uniform"] * self.k

            # Cria blocos contíguos de IDs por classe: [0..n0-1], [n0..n0+n1-1], ...
            self.nodes: dict[int, torch.Tensor] = {}
            for idx, size in enumerate(self.number_of_nodes):
                size = int(size)
                if idx == 0:
                    self.nodes[idx] = torch.arange(size, dtype=torch.int)
                else:
                    start = int(self.nodes[idx - 1][-1].item()) + 1
                    self.nodes[idx] = torch.arange(start, start + size, dtype=torch.int)

            # P(interclasse) — se não vier, usa uniforme
            if noise_class_distribution is None:
                self.noise_class_distribution = torch.full((self.k,), 1.0 / self.k)
            else:
                self.noise_class_distribution = noise_class_distribution

            # Estruturas
            self.pdf_distributions: dict[int, torch.Tensor] = {}
            self.edge_index = {c: torch.empty((2, 0), dtype=torch.long) for c in range(self.k)}  # intra
            self.node_degrees = torch.zeros(int(torch.sum(self.number_of_nodes)))
            self.computed_distributions: dict[int, torch.Tensor] = {}
            self.y = torch.tensor(np.concatenate([[c] * int(self.number_of_nodes[c]) for c in range(self.k)]))

            print(f"[UnAtt] Init (from scratch): k={self.k}, sizes={self.number_of_nodes.tolist()}, dists={self.edge_distributions}")

        else:
            # ---------------- MODO "MIMETIZAR" ----------------
            self.k = int(self.data_to_mimic.y.max().item()) + 1

            # Se number_of_nodes == 0 => usa as contagens originais por classe
            if number_of_nodes is 0:
                _, self.number_of_nodes = torch.unique(self.data_to_mimic.y, return_counts=True)

            # Cria blocos contíguos de IDs por classe
            self.nodes = {}
            for idx, size in enumerate(self.number_of_nodes):
                size = int(size)
                if idx == 0:
                    self.nodes[idx] = torch.arange(size, dtype=torch.int)
                else:
                    start = int(self.nodes[idx - 1][-1].item()) + 1
                    self.nodes[idx] = torch.arange(start, start + size, dtype=torch.int)

            # P(interclasse) proporcional às arestas interclasse por classe no original
            hetero_edges_per_class = list(count_heterogeneity(self.data_to_mimic.edge_index, self.data_to_mimic.y).values())
            self.noise_class_distribution = torch.tensor(hetero_edges_per_class, dtype=torch.float)
            self.noise_class_distribution = self.noise_class_distribution / self.noise_class_distribution.sum()

            # PDF por classe: graus do subgrafo da classe em representação undirected simétrica
            self.pdf_distributions = {}
            edge_index_by_label = split_edge_index_by_label(self.data_to_mimic.edge_index, self.data_to_mimic.y)

            # Fazer uma função de split_nodes_by_label

            ####################################################################################################
            # TODO: solve the problem of indexing
            for c in range(self.k):
                ei_c = remap_edge_index(edge_index_by_label[c])          # reindexa 0..|V_c|-1
                ei_c = simple_undirected(ei_c)                           # duplica (u,v)/(v,u), sem loops/dups
                # num_nodes_c = int(ei_c.max().item()) + 1 if ei_c.numel() > 0 else int((self.data_to_mimic.y == c).sum().item())
                num_nodes_c = self.number_of_nodes[c]

                # Adding one to avoid 0 degree nodes
                deg_c = degree(ei_c[0], num_nodes=num_nodes_c) + 1         # graus (não-dir) pois ei_c é simétrico
                s = deg_c.sum()
                if s > 0:
                    self.pdf_distributions[c] = deg_c / s
                else:
                    raise NameError('Graph with no edges')

            ####################################################################################################

            # Estruturas
            self.edge_index = {c: torch.empty((2, 0), dtype=torch.long) for c in range(self.k)}  # intra
            self.node_degrees = torch.zeros(int(torch.sum(self.number_of_nodes)))
            self.y = torch.tensor(np.concatenate([[c] * int(self.number_of_nodes[c]) for c in range(self.k)]))

            print(f"[UnAtt] Init (mimic): k={self.k}, sizes={self.number_of_nodes.tolist()}")

        # Placeholder de features (ajuste conforme sua aplicação)
        self.x = torch.randint(0, 1, size=(int(sum(self.number_of_nodes).item()), 2))

        # Conjuntos para evitar repetição de pares {u,v}
        self._intra_used_pairs: dict[int, set[tuple[int,int]]] = {c: set() for c in range(self.k)}
        self._global_used_pairs: set[tuple[int,int]] = set()

        # Grafo completo (será preenchido em edge_generation)
        self.whole_graph_edge_index = torch.empty((2, 0), dtype=torch.long)
        self.whole_graph_edge_index_unique = torch.empty((2, 0), dtype=torch.long)  # 1x por {u,v}


    # -------------------------------------------------------------------------
    # PDFs paramétricas (modo "do zero")
    # -------------------------------------------------------------------------
    def generate_distributions(
        self,
        mu: float = 0.5,
        sigma: float = 0.15,
        alpha: float = 1.2,
        lbd: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
    ):
        """Gera p(nó|classe) a partir de famílias indicadas em `edge_distributions`."""
        if self.data_to_mimic is not None:
            # No modo mimic já definimos as PDFs a partir do grafo original.
            return

        for c in range(self.k):
            x = np.linspace(0, 1, int(self.number_of_nodes[c]) + 1)[1:]  # evita f(0)=0
            name = self.edge_distributions[c]

            if name == "uniform":
                w = stats.uniform.pdf(x)
            elif name == "normal":
                w = stats.norm.pdf(x, mu, sigma)
            elif name == "powerlaw":
                w = stats.powerlaw.pdf(x, alpha)
            elif name == "Exponential":
                w = stats.expon.pdf(x, scale=alpha)
            elif name == "lognormal":
                w = stats.lognorm.pdf(x, s=sigma, scale=mu)
            elif name == "weibull":
                w = stats.weibull_min.pdf(x, c=a, scale=b)
            else:
                raise ValueError(f"Distribuição não suportada: {name}")

            w = w / w.sum()
            self.pdf_distributions[c] = torch.tensor(w, dtype=torch.float)


    # -------------------------------------------------------------------------
    # Gera 1 aresta INTRACLASSE evitando duplicatas
    # -------------------------------------------------------------------------
    def generate_edge(self, cluster: int):
        """
        Amostra {vi,vj} da classe 'cluster' conforme p(nó|classe) e adiciona a aresta
        nas duas direções (padrão PyG). Evita self-loop e repetição do par {vi,vj}.
        """
        while True:
            vi = self.nodes[cluster][torch.multinomial(self.pdf_distributions[cluster], 1, replacement=True)].item()
            vj = self.nodes[cluster][torch.multinomial(self.pdf_distributions[cluster], 1, replacement=True)].item()
            if vi == vj:
                continue
            key = (vi, vj) if vi < vj else (vj, vi)
            if key in self._intra_used_pairs[cluster]:
                continue
            self._intra_used_pairs[cluster].add(key)
            break

        new_edge = torch.tensor([[vi, vj], [vj, vi]], dtype=torch.long)          # 2 direções
        self.edge_index[cluster] = torch.cat([self.edge_index[cluster], new_edge], dim=1)

        self.node_degrees[vi] += 1
        self.node_degrees[vj] += 1


    # -------------------------------------------------------------------------
    # (Opcional) PDF empírica a partir dos graus gerados
    # -------------------------------------------------------------------------
    def compute_pdf_for_edge_index(self, cluster: int):
        """
        Recalcula p(nó|classe) a partir dos graus já gerados (útil como checagem).
        """
        if cluster == 0:
            sl = slice(0, int(self.number_of_nodes[0]))
        else:
            start = int(torch.sum(self.number_of_nodes[:cluster]).item())
            sl = slice(start, start + int(self.number_of_nodes[cluster]))
        degs = self.node_degrees[sl]
        s = degs.sum()
        self.computed_distributions[cluster] = degs / s if s > 0 else torch.full_like(degs, 1.0 / max(1, degs.numel()))


    # -------------------------------------------------------------------------
    # Gera 1 aresta INTERCLASSE evitando duplicatas globais
    # -------------------------------------------------------------------------
    def inter_community_edges(self, max_resamples: int = 50):
        """
        Amostra duas classes distintas conforme noise_class_distribution e conecta
        um nó de cada classe. Garante não repetir o par {vi,vj} no grafo global.
        """
        tried = 0
        while tried < max_resamples:
            cls_idx = torch.multinomial(self.noise_class_distribution, 2, replacement=False)
            Ci, Cj = int(cls_idx[0].item()), int(cls_idx[1].item())

            vi = self.nodes[Ci][torch.multinomial(self.pdf_distributions[Ci], 1, replacement=True)].item()
            vj = self.nodes[Cj][torch.multinomial(self.pdf_distributions[Cj], 1, replacement=True)].item()

            key = (vi, vj) if vi < vj else (vj, vi)
            if key not in self._global_used_pairs:
                self._global_used_pairs.add(key)
                new_edge = torch.tensor([[vi, vj], [vj, vi]], dtype=torch.long)
                self.whole_graph_edge_index = torch.cat([self.whole_graph_edge_index, new_edge], dim=1)
                return True
            tried += 1
        return False  # não conseguiu um par novo dentro do limite


    # -------------------------------------------------------------------------
    # Laço principal de geração
    # -------------------------------------------------------------------------
    def edge_generation(
        self,
        num_edges: list[int] | int = 0,   # nº de ARESTAS NÃO-DIRECIONADAS por classe
        interclass_edges: int = 0,        # nº de ARESTAS NÃO-DIRECIONADAS entre classes
        step: int = 50,
        verbose: bool = False,
    ):
        """
        Gera arestas intra e interclasse.

        Se data_to_mimic != None e num_edges == interclass_edges == 0:
          - Inferimos automaticamente as quantidades NÃO-DIRECIONADAS do original:
            * num_edges[c]  = |E_c| (intra, uma vez por {u,v})
            * interclass_edges = |E_inter| (uma vez por {u,v})
        """

        # PDFs paramétricas se estamos no modo "do zero"
        if self.data_to_mimic is None:
            self.generate_distributions()

        # Inferência automática das quantidades (modo mimic)
        if self.data_to_mimic is not None and num_edges is 0 and interclass_edges == 0:
            by_label = split_edge_index_by_label(self.data_to_mimic.edge_index, self.data_to_mimic.y)
            num_edges = [undirected_edge_count(by_label[c]) for c in range(self.k)]
            interclass_edges = interclass_undirected_edge_count(self.data_to_mimic.edge_index, self.data_to_mimic.y)

            if verbose:
                print("[UnAtt] Targets (undirected): intra per class =", num_edges,
                      "| interclass =", interclass_edges)

        # ---------- Intra ----------
        for c in range(self.k):
            # print('c',c)
            # print('num_edges', num_edges)
            # print('num_edges[c]', num_edges[c])
            target = int(num_edges[c]) if isinstance(num_edges, (list, tuple, torch.Tensor)) else int(num_edges)
            print(f'{target} edges to be added to cluster {c}')
            for i in range(target):
                if verbose and (i % max(1, step) == 0):
                    print(f"[UnAtt] Intra cluster {c}: {i}/{target}", end="\r")
                self.generate_edge(c)
            print()

        # Monta o grafo completo a partir dos intra
        self.whole_graph_edge_index = torch.empty((2, 0), dtype=torch.long)
        for c in range(self.k):
            self.whole_graph_edge_index = torch.cat([self.whole_graph_edge_index, self.edge_index[c]], dim=1)

        # Inicializa o conjunto global com os pares intra já usados
        for c in range(self.k):
            self._global_used_pairs.update(self._intra_used_pairs[c])

        # ---------- Inter ----------
        added = 0
        attempts = 0
        # Garante gerar exatamente `interclass_edges` arestas únicas (com limite de tentativas)
        while added < int(interclass_edges) and attempts < int(interclass_edges) * 20:
            ok = self.inter_community_edges()
            added += int(ok)
            attempts += 1
            if verbose and (added % max(1, step) == 0):
                print(f"[UnAtt] Inter-class: {added}/{interclass_edges}", end="\r")

        # Representações finais para contagem/validação
        self.whole_graph_edge_index_unique = unique_undirected_edges(self.whole_graph_edge_index)
        # (se quiser manter um edge_index simétrico limpo:)
        self.whole_graph_edge_index = simple_undirected(self.whole_graph_edge_index)