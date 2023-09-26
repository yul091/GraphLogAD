import math
import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
from torch_geometric.data import Data
from .graph_base import nn, torch, Tensor


class SCAN(nn.Module):
    """
    Parameters
    ----------
    eps : float, optional
        Neighborhood threshold. Default: ``.5``.
    mu : int, optional
        Minimal size of clusters. Default: ``2``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.
    """

    def __init__(self, eps: float = 0.5, mu: int = 2, contamination: float = 0.1, verbose: bool = False):
        super().__init__()
        # model param
        self.eps = eps
        self.mu = mu
        self.contamination = contamination
        self.verbose = verbose


    def forward(self, G: Data) -> Tensor:
        decision_scores = np.zeros(G.x.shape[0])
        G = self.process_graph(G) # convert to networkx graph
        c = 0
        clusters = {}
        nomembers = []
        for n in G.nodes():
            if self.hasLabel(clusters, n):
                continue
            else:
                N = self.neighborhood(G, n)
                if len(N) > self.mu:
                    c = c + 1
                    Q = self.neighborhood(G, n)
                    clusters[c] = []
                    clusters[c].append(n) # append core vertex itself
                    while len(Q) != 0:
                        w = Q.pop(0)
                        R = self.neighborhood(G, w)
                        R.append(w) # include current vertex itself
                        for s in R:
                            if not (self.hasLabel(clusters, s)) or s in nomembers:
                                clusters[c].append(s)
                            if not (self.hasLabel(clusters, s)):
                                Q.append(s)
                else:
                    nomembers.append(n)
        
        for k, v in clusters.items():
            decision_scores[v] = 1
        return torch.FloatTensor(decision_scores)     
        
    def neighborhood(self, G: Graph, v: int):
        eps_neighbors = []
        v_list = G.neighbors(v)
        for u in v_list:
            if (self.similarity(G, u, v)) > self.eps:
                eps_neighbors.append(u)
        return eps_neighbors
    
    def similarity(self, G: Graph, v: int, u: int):
        v_set = set(G.neighbors(v))
        u_set = set(G.neighbors(u))
        inter = v_set.intersection(u_set)
        if inter == 0:
            return 0
        # Need to account for vertex itself, add 2(1 for each vertex)
        sim = (len(inter) + 2) / (math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))
        return sim

    def hasLabel(self, cliques: dict, vertex: int):
        for k, v in cliques.items():
            if vertex in v:
                return True
        return False

    def sameClusters(self, G: Graph, clusters: dict, u: int):
        n = G.neighbors(u)
        b = []
        i = 0
        while i < len(n):
            for k, v in clusters.items():
                if n[i] in v:
                    if k in b:
                        continue
                    else:
                        b.append(k)
            i = i + 1
        if len(b) > 1:
            return False
        return True

    def process_graph(self, G: Data) -> Graph:
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        G : networkx.classes.graph.Graph
            NetworkX Graph
        """
        G = nx.from_edgelist(G.edge_index.T.tolist())
        return G