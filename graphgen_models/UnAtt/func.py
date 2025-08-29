import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    degree,
    to_undirected,
    coalesce,
    remove_self_loops,
)

def split_edge_index_by_label(edge_index: torch.Tensor, y: torch.Tensor):
    """
    Separa o edge_index em subgrafos por label da classe dos vértices.
    
    Parameters
    ----------
    edge_index : torch.Tensor
        Tensor [2, E] com as arestas (no estilo PyG).
    y : torch.Tensor
        Tensor [N] com os rótulos de cada nó.
    
    Returns
    -------
    dict[int, torch.Tensor]
        Dicionário {label: edge_index_label}, onde edge_index_label contém
        apenas arestas conectando nós dessa classe.
    """
    # rótulo de cada nó para cada extremidade da aresta
    src, dst = edge_index
    label_src = y[src]
    label_dst = y[dst]

    # máscara: só pega arestas com ambos extremos da mesma classe
    same_label_mask = label_src == label_dst

    # arestas filtradas
    filtered_edges = edge_index[:, same_label_mask]

    # cria o dicionário
    result = {}
    for lbl in torch.unique(y):
        mask_lbl = (label_src == lbl) & (label_dst == lbl)
        edges_lbl = edge_index[:, mask_lbl]
        result[int(lbl.item())] = edges_lbl

    return result

def count_heterogeneity(edge_index: torch.Tensor, y: torch.Tensor):
    """
    Para cada classe c, conta as arestas (src->dst) em que y[src] = c e y[dst] != c.
    Observação: se o grafo for não-direcionado (com arestas duplicadas i->j e j->i),
    cada aresta heterogênea contribui 1 para cada direção.
    """
    src, dst = edge_index
    mask_diff = (y[src] != y[dst])           # só arestas entre classes distintas
    counts = torch.bincount(y[src][mask_diff],
                            minlength=int(y.max().item())+1)
    labels = torch.unique(y).tolist()
    return {int(lbl): int(counts[int(lbl)].item()) for lbl in labels}

def remap_edge_index(edge_index: torch.Tensor):
    # Pega todos os nós que aparecem no edge_index
    unique_nodes = torch.unique(edge_index)
    
    # Cria um mapeamento {id_original -> id_remapeado}
    mapping = {int(node): i for i, node in enumerate(unique_nodes.tolist())}
    
    # Aplica o mapeamento no edge_index
    remapped = torch.tensor([[mapping[int(node)] for node in row] for row in edge_index], 
                            dtype=torch.long)
    
    return remapped


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