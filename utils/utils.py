import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
import igraph as ig
from scipy import sparse
import json
import random
import argparse
import yaml
from torch_geometric import datasets
from torch_geometric.datasets.graph_generator import BAGraph
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
from graphgen_models.SkyMap.main import SkyMap
from graphgen_models.GenCAT import func
from graphgen_models.GenCAT import gencat
from scipy.sparse import csr_matrix
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from graphgen_models.UnAtt.model import unatt
from graphgen_models.UnAtt import func as unnat_utils
import os


def parse_arguments():
    '''
    Function to collect the arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config_file', type=str, help = 'configuration file .yaml')
    
    return parser.parse_args()

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def generate_class_colors(k):
    """
    Gera um dicionário onde a chave é a classe (de 0 a k-1) e o valor é uma cor associada a essa classe.
    
    Args:
        k (int): Número de classes.
    
    Returns:
        dict: Dicionário de cores, com chave=classe e valor=cor.
    """
    # Gerar k cores distintas usando uma paleta de cores de 'tab10' do matplotlib
    colors_map = plt.cm.get_cmap('tab10', k)  # 'tab10' é uma paleta de 10 cores distintas
    
    # Criar o dicionário de classes:cores
    class_colors = {i: colors_map(i) for i in range(k)}
    
    return class_colors

def sparse_to_igraph(S):
    """
    Converte uma matriz esparsa (CSR ou COO) em um grafo não direcionado no formato igraph.
    
    Args:
    S (scipy.sparse.csr_matrix or scipy.sparse.coo_matrix): A matriz esparsa (n x n) do grafo.
    
    Returns:
    g (igraph.Graph): O grafo no formato igraph.
    """
    # Converte a matriz esparsa para COO (se não for o formato desejado)
    S_coo = sparse.coo_matrix(S)
    
    # Número de nós
    n = S.shape[0]
    
    # Cria o grafo não direcionado
    g = ig.Graph()
    g.add_vertices(n)
    
    # Adiciona arestas (evitando duplicatas)
    edges = set()  # Usamos um set para garantir que arestas duplicadas não sejam adicionadas
    for row, col in zip(S_coo.row, S_coo.col):
        if row != col:  # Evita adicionar laços (self-loops)
            # Adiciona aresta em ordem (min(row, col), max(row, col)) para garantir que (i,j) == (j,i)
            edges.add(tuple(sorted([row, col])))
    
    # Adiciona as arestas ao grafo
    g.add_edges(list(edges))
    
    return g

def edge_index_to_igraph(edge_index):
    """
    Converte um edge_index do PyTorch Geometric para um grafo do igraph.

    Args:
        edge_index (torch.Tensor): Tensor de shape [2, E] representando as arestas.
        
    Returns:
        igraph.Graph: Grafo criado a partir do edge_index.
    """
    # Verifica se o edge_index tem o formato correto
    assert edge_index.size(0) == 2, "O edge_index deve ter 2 linhas (origem e destino das arestas)."
    
    # Converte o edge_index para uma lista de tuplas (origem, destino)
    edges = edge_index.t().tolist()
    
    # Cria o grafo a partir das arestas
    g = ig.Graph(edges=edges)
    
    return g

def get_data(config):
    '''
    Returns the dataset file
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        torch_geometric.data.Data: Graph data file
    '''

    # Verify if datasets folder exists

    datasets_path = "./datasets/"

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
    
    dataset_class = config['sfanalysis']['dataset_name'].split(":")[0]

    try:
        dataset_name = config['sfanalysis']['dataset_name'].split(":")[1]
    except:
        dataset_name = None
    
    if dataset_class == "AttributedGraphDataset":
        data = datasets.AttributedGraphDataset(root = 'datasets', name = dataset_name)[0]
        
    if dataset_class == "Coauthor":
        data = datasets.Coauthor(root = 'datasets', name = dataset_name)[0]

    if dataset_class == "Amazon":
        data = datasets.Amazon(root = 'datasets', name = dataset_name)[0]
    
    if dataset_class == "Planetoid":
        data = datasets.Planetoid(root = 'datasets', name = dataset_name)[0]

    if dataset_class == "Coauthor":
        data = datasets.Coauthor(root = 'datasets', name = dataset_name)[0]

    if dataset_class == 'CitationFull':
        data = datasets.CitationFull(root = 'datasets', name = dataset_name)

    if dataset_class == 'WikiCS':
        data = datasets.WikiCS(root = 'datasets')

    if dataset_class == 'KarateClub':
        data = datasets.KarateClub()

    if config['sfanalysis']['mimic_data']:
        print(f'mimicking data with values n {config["sfanalysis"]["n"]}, m {config["sfanalysis"]["m"]}')

        if config['sfanalysis']['synthetic_gen'] == 'GenCAT':

            # Create the npz file with utils func
            edge_index_to_npz_file(edge_index = data.edge_index.detach().numpy(), x = data.x.detach().numpy(), y = data.y.detach().numpy())

            S_ori, C, _, _, k = func.load_data('datasets/tmp_graph.npz')
            

            # Creating a random H matrix, it is useless for Scale-free analysis
            d = 2
            H = np.random.rand(d,k)

            n = config['sfanalysis']['n'] * len(data.y)
            m = config['sfanalysis']['m'] * len(data.edge_index[0] / 2)
            maxdeg = 0

            if config['sfanalysis']['n'] != 0 and config['sfanalysis']['m'] != 0:

                theta = np.zeros(len(C))
                nnz = S_ori.nonzero()
                for i in range(len(nnz[0])):
                    if nnz[0][i] < nnz[1][i]:
                        theta[nnz[0][i]] += 1
                        theta[nnz[1][i]] += 1
                maxdeg = max(theta)

            S,X,Label = gencat.gencat_reproduction(S_ori, C, H, d, n=n,m=m,max_deg=int(maxdeg*2))

            print(type(S), type(X))

            edge_index, _ = dok_to_edge_index_pyg(S, undirected = True)

            aug_data = Data(edge_index = edge_index, x = torch.tensor(X), y = torch.tensor(Label))

            return aug_data

        if config['sfanalysis']['synthetic_gen'] == 'SkyMap':
            skymap = SkyMap()
            input_graph = to_networkx(data, to_undirected=True)

            nx.set_node_attributes(input_graph, dict(zip(input_graph.nodes, data.y.tolist())), "y")
            nx.set_node_attributes(input_graph, dict(zip(input_graph.nodes, data.x.tolist())), "x")


            n = config['sfanalysis']['n'] * len(data.y)
            m = config['sfanalysis']['m'] * len(data.edge_index[0] / 2)

            if config['sfanalysis']['n'] == 0:
                n = None

            skymap_graph = skymap.mimic_graph(input_graph, num_nodes=n)

            aug_data = from_networkx(skymap_graph)

            return aug_data
        
        if config['sfanalysis']['synthetic_gen'] == 'UnAtt':
            n = config['sfanalysis']['n'] * torch.unique(data.y, return_counts = True)[1]
            m = list(unnat_utils.split_edge_index_by_label(data.edge_index, data.y).values())
            print(n,m)
            m = config['sfanalysis']['m'] * m
            intra_cluster_edges = sum(list(unnat_utils.count_heterogeneity(data.edge_index, data.y).values())) * config['sfanalysis']['intra_cluster_edges']

            if config['sfanalysis']['n'] == 0:
                n = 0
            if config['sfanalysis']['m']  == 0:
                m = 0
            if  config['sfanalysis']['intra_cluster_edges'] == 0:
                intra_cluster_edges = 0

            aug_data = unatt(data_to_mimic = data, number_of_nodes = n)
            aug_data.edge_generation(num_edges = m, interclass_edges = intra_cluster_edges)
            aug_data.edge_index = aug_data.whole_graph_edge_index

            return aug_data


    if not config['sfanalysis']['mimic_data'] and config['sfanalysis']['dataset_name'] == None:
        if config['sfanalysis']['synthetic_gen'] == 'BA':
            # num_nodes and num_edges should be in the interval 1000, 6 ~ 9
            num_nodes = 5000
            # Cada novo nó conecta a m nós existentes
            num_edges = 15

            BA_graph = nx.barabasi_albert_graph(num_nodes, num_edges)
            data = from_networkx(BA_graph)

            data.x = torch.eye(num_nodes)
            data.y = torch.zeros(num_nodes)

        if dataset_name == 'ALBTER':
            print()
       
    return data

def count_hetero_edges(edge_index: torch.Tensor, y: torch.Tensor):
    """
    Conta, para cada classe c em y, quantas arestas do grafo conectam nós da
    classe c a nós de classes diferentes (heterofilia).

    Parâmetros
    ----------
    edge_index : torch.Tensor
        Tensor shape (2, E) com as arestas do grafo. Pode estar direcionado
        (i.e., conter (u,v) e (v,u)) e/ou conter auto-laços.
    y : torch.Tensor
        Tensor shape (N,) com o rótulo inteiro de cada nó.

    Retorno
    -------
    Dict[int, int]
        Dicionário {classe -> número de arestas hetero incidentes à classe}.
        Observação: cada aresta hetero contribui +1 para cada uma das duas
        classes envolvidas.
    """
    if not (isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2):
        raise ValueError("edge_index deve ser um tensor (2, E).")
    if not (isinstance(y, torch.Tensor) and y.dim() == 1):
        raise ValueError("y deve ser um tensor (N,) com rótulos inteiros por nó.")

    device = edge_index.device
    y = y.to(device).to(torch.long)

    # Remove auto-laços
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    src, dst = src[mask], dst[mask]

    # Colapsa para arestas não dirigidas únicas: (min(u,v), max(u,v))
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    undirected = torch.stack([u, v], dim=0)                # (2, E')
    undirected = torch.unique(undirected, dim=1)           # remove duplicadas coluna a coluna
    u, v = undirected[0], undirected[1]

    # Classes presentes
    classes = torch.unique(y).tolist()

    # Rótulos dos endpoints
    yu = y[u]
    yv = y[v]

    # Arestas hetero: rótulos diferentes
    hetero_mask = yu != yv
    u_h = u[hetero_mask]
    v_h = v[hetero_mask]
    yu_h = y[u_h]
    yv_h = y[v_h]

    # Contagem por classe:
    # cada aresta hetero (c1, c2) conta +1 para c1 e +1 para c2
    counts = {int(c): 0 for c in classes}
    # Contabiliza extremos esquerdos
    if yu_h.numel() > 0:
        uniq, freq = torch.unique(yu_h, return_counts=True)
        for c, f in zip(uniq.tolist(), freq.tolist()):
            counts[int(c)] += int(f)
    # Contabiliza extremos direitos
    if yv_h.numel() > 0:
        uniq, freq = torch.unique(yv_h, return_counts=True)
        for c, f in zip(uniq.tolist(), freq.tolist()):
            counts[int(c)] += int(f)

    return counts

def undirected_edge_count(edge_index: torch.Tensor) -> int:
    """Conta arestas únicas não dirigidas (ignora auto-laços e duplicatas)."""
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    u = torch.minimum(src[mask], dst[mask])
    v = torch.maximum(src[mask], dst[mask])
    undirected = torch.unique(torch.stack([u, v], dim=0), dim=1)
    return undirected.size(1)

def edge_index_to_npz_file(
    edge_index,           # np.ndarray shape (2, E) com arestas (u,v)
    x,                    # np.ndarray shape (N, F) atributos dos nós (denso ou esparso-coo)
    y,                    # np.ndarray shape (N,) rótulos inteiros
    idx_to_node=None,     # dict {i: node_id_str} opcional
    idx_to_attr=None,     # dict {j: attr_name} opcional
    idx_to_class=None,    # dict {k: class_name} opcional
    texts=None,           # list[str] tamanho N (opcional)
    undirected=True,
    out_path="datasets/tmp_graph.npz"
):
    N = int(x.shape[0])
    # --- Adjacência ---
    u = edge_index[0].astype(np.int64)
    v = edge_index[1].astype(np.int64)

    if undirected:
        # garante simetria
        u = np.concatenate([u, v])
        v = np.concatenate([v, u[:len(v)]])  # cuidado para não bagunçar; outra forma:
        # uma forma mais clara:
        # u = np.concatenate([edge_index[0], edge_index[1]])
        # v = np.concatenate([edge_index[1], edge_index[0]])

    # remove self-loops e duplicatas
    mask = u != v
    u, v = u[mask], v[mask]
    edges = np.vstack([np.minimum(u, v), np.maximum(u, v)]).T
    edges = np.unique(edges, axis=0)

    # constrói CSR da adjacência binária
    data = np.ones(edges.shape[0], dtype=np.float64)
    A = csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=(N, N))
    if undirected:
        A = A.maximum(A.T)

    # --- Atributos ---
    if isinstance(x, csr_matrix):
        X = x.tocsr().astype(np.float64)
    else:
        X = csr_matrix(x.astype(np.float64))  # esparso, se x for denso

    # --- Rótulos ---
    labels = y.astype(np.int64)

    # --- Metadados/dicionários ---
    if idx_to_node is None:
        idx_to_node = {i: i for i in range(N)}
    if idx_to_attr is None:
        idx_to_attr = {j: j for j in range(X.shape[1])}
    if idx_to_class is None:
        classes = np.unique(labels)
        idx_to_class = {int(c): f"class_{int(c)}" for c in classes}
    if texts is None:
        texts = np.array([""] * N, dtype=np.str_)  # pode colocar strings reais se tiver

    # --- Pacote no formato do cora_ml.npz ---
    payload = {
        # Adjacência CSR
        "adj_data":    A.data.astype(np.float64),
        "adj_indices": A.indices.astype(np.int32),
        "adj_indptr":  A.indptr.astype(np.int32),
        "adj_shape":   np.array(A.shape, dtype=np.int64),

        # Atributos CSR
        "attr_data":    X.data.astype(np.float64),
        "attr_indices": X.indices.astype(np.int32),
        "attr_indptr":  X.indptr.astype(np.int32),
        "attr_shape":   np.array(X.shape, dtype=np.int64),

        # Rótulos
        "labels": labels,

        # Textos e dicionários (armazenados como objetos)
        "attr_text":   np.array(texts, dtype=np.str_),
        "idx_to_node": np.array(idx_to_node, dtype=object),
        "idx_to_attr": np.array(idx_to_attr, dtype=object),
        "idx_to_class": np.array(idx_to_class, dtype=object),
    }

    # Salvando no mesmo “estilo”: 1 objeto (dict) em arr_0
    np.savez(out_path, payload)
    print(f"Salvo em: {out_path}")

def dok_to_edge_index_pyg(A_dok, undirected=False, remove_self_loops=True, device=None):
    # DOK -> CSR (mais eficiente)
    A = A_dok.tocsr()

    if undirected:
        # simetriza mantendo o máximo (ou soma, se preferir A + A.T)
        A = A.maximum(A.T)

    if remove_self_loops:
        A.setdiag(0)

    A.eliminate_zeros()   # remove zeros explícitos
    A.sum_duplicates()    # consolida duplicatas

    edge_index, edge_weight = from_scipy_sparse_matrix(A)  # tensors long/float
    if device is not None:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    return edge_index, edge_weight

def edge_index_to_gml(edge_index, path, directed=False):
    # edge_index: Tensor shape [2, E]
    ei = edge_index.detach().cpu().numpy()
    G = nx.Graph() if not directed else nx.DiGraph()
    G.add_edges_from(zip(ei[0], ei[1]))
    # garante não-direcionado sem duplicatas
    if directed is False:
        G = nx.Graph(G)
    # nx.write_gml(G, path)
    nx.write_gml(G, path, stringizer=str)
    return path

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