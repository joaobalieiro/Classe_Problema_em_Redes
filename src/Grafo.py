import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, colors
from scipy.sparse import csr_matrix

class Grafo:
    def __init__(self, nodes=None, edges=None, kind=None):
        """
        Parameters:
        - nodes: list of node identifiers or dict of node_id -> properties
        - edges: dict of source_node -> dict of target_node -> properties
        - kind: "Directed" or "Undirected" (default: "Undirected")
        """
        if isinstance(nodes, list):
            self.nodes = {n: {} for n in nodes}  
        elif isinstance(nodes, dict):
            self.nodes = nodes
        elif nodes is None:
            self.nodes = {}
        else:
            raise TypeError("nodes must be a list or dict")
        self.edges = edges if edges is not None else {}
        self.kind = "Undirected" if kind is None else kind

        self._validate()

        self.num_nodes = len(self.nodes)
        self.num_edges = sum(len(link) for link in self.edges.values())

    def _validate(self):
        """Checks if all edges connect existing nodes."""
        for u in self.edges:
            if u not in self.nodes:
                raise Exception(f"Source node {u} not found.")

            for v in self.edges[u]:
                if v not in self.nodes:
                    raise Exception(f"Target node {v} not found.")

                if self.kind == 'Directed':
                    if v in self.edges and u in self.edges[v]:
                        raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")
        return True

    def insert_node(self, index, **kwargs):
        if index not in self.nodes:
            self.nodes[index] = kwargs
            self.edges[index] = {}
            self.num_nodes += 1

    def insert_edge(self, u, v, **kwargs):
        if u not in self.nodes or v not in self.nodes:
            raise Exception(f"Nodes {u} and {v} must exist.")

        if self.kind == 'Directed':
            if v in self.edges and u in self.edges[v]:
                raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")

        if u not in self.edges:
            self.edges[u] = {}

        if v not in self.edges[u]:
            self.num_edges += 1

        self.edges[u][v] = kwargs

    def get_edge_data(self, u, v):
        return self.edges.get(u, {}).get(v, None)

    def remove_node(self, index):
        if index not in self.nodes:
            raise Exception(f"Node {index} not found.")

        # remover arestas saindo do nó
        if index in self.edges:
            self.num_edges -= len(self.edges[index])
            del self.edges[index]

        # remover arestas entrando no nó
        for u in self.edges:
            if index in self.edges[u]:
                del self.edges[u][index]
                self.num_edges -= 1

        del self.nodes[index]
        self.num_nodes -= 1
    
    def get_node_order(self):
        return list(self.nodes.keys())

    def get_edge_list(self):
        return [(u, v) for u in self.edges for v in self.edges[u]]

    def compute_connection_matrix(self, sparse_output=True):
        """
        Returns the incidence matrix for a directed graph.

        Each row represents an edge.
        +1 -> source node
        -1 -> target node
        """

        if self.kind != "Directed":
            raise ValueError("Connection matrix is defined only for directed graphs.")

        node_list = self.get_node_order()
        node_index = {node: i for i, node in enumerate(node_list)}
        edge_list = self.get_edge_list()

        rows = []
        cols = []
        data = []

        for row, (u, v) in enumerate(edge_list):
            rows.extend([row, row])
            cols.extend([node_index[u], node_index[v]])
            data.extend([1.0, -1.0])

        B = csr_matrix(
            (data, (rows, cols)),
            shape=(len(edge_list), len(node_list)),
            dtype=float
        )

        if sparse_output:
            return B
        return B.toarray()

    def get_network(self):
        G = nx.DiGraph() if self.kind == "Directed" else nx.Graph()

        # Adiciona os nós e suas propriedades
        for u, attrs in self.nodes.items():
            G.add_node(u, **attrs)

        # Adiciona as arestas
        for u in self.edges:
            for v, attrs in self.edges[u].items():
                G.add_edge(u, v, **attrs)

        return G

    def _get_layout_positions(self, G, layout, seed=42):
        if layout == "planar":
            try:
                return nx.planar_layout(G)
            except Exception:
                return nx.spring_layout(G, seed=seed)

        if layout == "spring":
            return nx.spring_layout(G, seed=seed)

        if layout == "circular":
            return nx.circular_layout(G)

        if layout == "shell":
            return nx.shell_layout(G)

        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)

        if layout == "spectral":
            return nx.spectral_layout(G)

        raise ValueError(
            "layout must be one of: 'planar', 'spring', 'circular', 'shell', 'kamada_kawai', 'spectral'"
        )

    def _format_attr_value(self, key, value, precision):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{key[0].upper()}:{float(value):.{precision}f}"
        return f"{key[0].upper()}:{value}"
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, colors
from scipy.sparse import csr_matrix

class Grafo:
    def __init__(self, nodes=None, edges=None, kind=None):
        """
        Parameters:
        - nodes: list of node identifiers or dict of node_id -> properties
        - edges: dict of source_node -> dict of target_node -> properties
        - kind: "Directed" or "Undirected" (default: "Undirected")
        """
        if isinstance(nodes, list):
            self.nodes = {n: {} for n in nodes}  
        elif isinstance(nodes, dict):
            self.nodes = nodes
        elif nodes is None:
            self.nodes = {}
        else:
            raise TypeError("nodes must be a list or dict")
        self.edges = edges if edges is not None else {}
        self.kind = "Undirected" if kind is None else kind

        self._validate()

        self.num_nodes = len(self.nodes)
        self.num_edges = sum(len(link) for link in self.edges.values())

    def _validate(self):
        """Checks if all edges connect existing nodes."""
        for u in self.edges:
            if u not in self.nodes:
                raise Exception(f"Source node {u} not found.")

            for v in self.edges[u]:
                if v not in self.nodes:
                    raise Exception(f"Target node {v} not found.")

                if self.kind == 'Directed':
                    if v in self.edges and u in self.edges[v]:
                        raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")
        return True

    def insert_node(self, index, **kwargs):
        if index not in self.nodes:
            self.nodes[index] = kwargs
            self.edges[index] = {}
            self.num_nodes += 1

    def insert_edge(self, u, v, **kwargs):
        if u not in self.nodes or v not in self.nodes:
            raise Exception(f"Nodes {u} and {v} must exist.")

        if self.kind == 'Directed':
            if v in self.edges and u in self.edges[v]:
                raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")

        if u not in self.edges:
            self.edges[u] = {}

        if v not in self.edges[u]:
            self.num_edges += 1

        self.edges[u][v] = kwargs

    def get_edge_data(self, u, v):
        return self.edges.get(u, {}).get(v, None)

    def remove_node(self, index):
        if index not in self.nodes:
            raise Exception(f"Node {index} not found.")

        # remover arestas saindo do nó
        if index in self.edges:
            self.num_edges -= len(self.edges[index])
            del self.edges[index]

        # remover arestas entrando no nó
        for u in self.edges:
            if index in self.edges[u]:
                del self.edges[u][index]
                self.num_edges -= 1

        del self.nodes[index]
        self.num_nodes -= 1
    
    def get_node_order(self):
        return list(self.nodes.keys())

    def get_edge_list(self):
        return [(u, v) for u in self.edges for v in self.edges[u]]

    def compute_connection_matrix(self, sparse_output=True):
        """
        Returns the incidence matrix for a directed graph.

        Each row represents an edge.
        +1 -> source node
        -1 -> target node
        """

        if self.kind != "Directed":
            raise ValueError("Connection matrix is defined only for directed graphs.")

        node_list = self.get_node_order()
        node_index = {node: i for i, node in enumerate(node_list)}
        edge_list = self.get_edge_list()

        rows = []
        cols = []
        data = []

        for row, (u, v) in enumerate(edge_list):
            rows.extend([row, row])
            cols.extend([node_index[u], node_index[v]])
            data.extend([1.0, -1.0])

        B = csr_matrix(
            (data, (rows, cols)),
            shape=(len(edge_list), len(node_list)),
            dtype=float
        )

        if sparse_output:
            return B
        return B.toarray()

    def get_network(self):
        G = nx.DiGraph() if self.kind == "Directed" else nx.Graph()

        # Adiciona os nós e suas propriedades
        for u, attrs in self.nodes.items():
            G.add_node(u, **attrs)

        # Adiciona as arestas
        for u in self.edges:
            for v, attrs in self.edges[u].items():
                G.add_edge(u, v, **attrs)

        return G

    def _get_layout_positions(self, G, layout, seed=42):
        if layout == "planar":
            try:
                return nx.planar_layout(G)
            except Exception:
                return nx.spring_layout(G, seed=seed)

        if layout == "spring":
            return nx.spring_layout(G, seed=seed)

        if layout == "circular":
            return nx.circular_layout(G)

        if layout == "shell":
            return nx.shell_layout(G)

        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)

        if layout == "spectral":
            return nx.spectral_layout(G)

        raise ValueError(
            "layout must be one of: 'planar', 'spring', 'circular', 'shell', 'kamada_kawai', 'spectral'"
        )

    def _format_attr_value(self, key, value, precision):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{key[0].upper()}:{float(value):.{precision}f}"
        return f"{key[0].upper()}:{value}"

    def plot(
        self,
        show_node_labels=True,
        show_edge_labels=True,
        precision=2,
        layout="planar",
        figsize=(10, 8),
        node_value_attr="pressao",
        edge_value_attr="vazao",
        node_cmap="viridis",
        edge_cmap="plasma",
        use_abs_edge_color=True,
        edge_width_range=(1.5, 5.5),
    ):

        G = self.get_network()
        pos = self._get_layout_positions(G, layout=layout)
        fig, ax = plt.subplots(figsize=figsize)

        # Lista dos nós e arestas na ordem em que serão processados
        node_order = list(G.nodes())
        edge_order = list(G.edges())

        # Extrai os valores numéricos dos atributos dos nós
        # Se o atributo não for numérico, usa NaN
        node_values = np.array(
            [
                float(G.nodes[node].get(node_value_attr, np.nan))
                if isinstance(G.nodes[node].get(node_value_attr, np.nan), (int, float, np.integer, np.floating))
                else np.nan
                for node in node_order
            ],
            dtype=float,
        )

        # Extrai os valores numéricos dos atributos das arestas
        # Se o atributo não for numérico, usa NaN
        edge_raw_values = np.array(
            [
                float(G.edges[edge].get(edge_value_attr, np.nan))
                if isinstance(G.edges[edge].get(edge_value_attr, np.nan), (int, float, np.integer, np.floating))
                else np.nan
                for edge in edge_order
            ],
            dtype=float,
        )

        # Se solicitado, usa o valor absoluto para colorir as arestas
        edge_color_values = np.abs(edge_raw_values) if use_abs_edge_color else edge_raw_values

        # Verifica se existem valores válidos para criar gradiente de cor
        has_node_gradient = np.isfinite(node_values).any()
        has_edge_gradient = np.isfinite(edge_color_values).any()

        # -------------------------
        # Desenho dos nós
        # -------------------------
        if has_node_gradient:
            # Considera apenas os valores finitos para normalização
            valid = node_values[np.isfinite(node_values)]
            vmin, vmax = float(np.min(valid)), float(np.max(valid))

            # Evita erro quando todos os valores são iguais
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-12

            # Normalização usada na colorbar
            node_norm = colors.Normalize(vmin=vmin, vmax=vmax)

            # Desenha os nós com gradiente de cor
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=900,
                node_color=node_values,
                cmap=cm.get_cmap(node_cmap),
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
        else:
            # Se não houver valores numéricos, usa uma cor fixa
            node_norm = None
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=900,
                node_color="skyblue",
                ax=ax,
            )

        # -------------------------
        # Desenho das arestas
        # -------------------------
        if has_edge_gradient:
            # Considera apenas os valores finitos para normalização
            valid = edge_color_values[np.isfinite(edge_color_values)]
            vmin, vmax = float(np.min(valid)), float(np.max(valid))

            # Evita erro quando todos os valores são iguais
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-12

            # Normalização para a colorbar das arestas
            edge_norm = colors.Normalize(vmin=vmin, vmax=vmax)

            # Substitui NaN por vmin para evitar problemas ao calcular espessuras
            finite_values = np.where(np.isfinite(edge_color_values), edge_color_values, vmin)

            # Define a espessura das arestas proporcionalmente ao valor
            if np.isclose(finite_values.max(), finite_values.min()):
                edge_widths = np.full(len(edge_order), float(np.mean(edge_width_range)))
            else:
                edge_widths = np.interp(
                    finite_values,
                    (finite_values.min(), finite_values.max()),
                    edge_width_range,
                )

            # Desenha as arestas com cor variando conforme o valor
            nx.draw_networkx_edges(
                G,
                pos,
                edge_color=edge_color_values,
                edge_cmap=cm.get_cmap(edge_cmap),
                edge_vmin=vmin,
                edge_vmax=vmax,
                width=edge_widths,
                arrows=self.kind == "Directed",
                arrowstyle="-|>" if self.kind == "Directed" else "-",
                arrowsize=20,
                connectionstyle="arc3,rad=0.03" if self.kind == "Directed" else "arc3",
                ax=ax,
            )
        else:
            # Se não houver valores válidos, desenha arestas com cor e espessura fixas
            edge_norm = None
            nx.draw_networkx_edges(
                G,
                pos,
                width=2.5,
                edge_color="dimgray",
                arrows=self.kind == "Directed",
                arrowstyle="-|>" if self.kind == "Directed" else "-",
                arrowsize=20,
                connectionstyle="arc3,rad=0.03" if self.kind == "Directed" else "arc3",
                ax=ax,
            )

        # -------------------------
        # Rótulos dos nós
        # -------------------------
        if show_node_labels:
            # Cria o texto dos rótulos com o nome do nó e seus atributos
            node_labels = {
                node: node + "\n" + "\n".join(
                    [self._format_attr_value(k, val, precision) for k, val in attr.items()]
                )
                if attr else node
                for node, attr in G.nodes(data=True)
            }

            # Desenha os rótulos dos nós
            nx.draw_networkx_labels(
                G,
                pos,
                labels=node_labels,
                font_color="black",
                font_size=9,
                ax=ax,
            )

    
        if show_edge_labels:
            edge_labels = {
                (u, v): "\n".join(
                    [self._format_attr_value(k, val, precision) for k, val in data.items()]
                )
                for u, v, data in G.edges(data=True)
                if data
            }

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_color="black",
                font_size=8,
                rotate=False,
                ax=ax,
            )

        if has_node_gradient:
            node_sm = cm.ScalarMappable(norm=node_norm, cmap=cm.get_cmap(node_cmap))
            node_sm.set_array([])
            cbar_nodes = fig.colorbar(node_sm, ax=ax, fraction=0.046, pad=0.04)
            cbar_nodes.set_label(node_value_attr.capitalize())


        if has_edge_gradient:
            edge_sm = cm.ScalarMappable(norm=edge_norm, cmap=cm.get_cmap(edge_cmap))
            edge_sm.set_array([])
            cbar_edges = fig.colorbar(edge_sm, ax=ax, fraction=0.046, pad=0.10)
            label = f"|{edge_value_attr}|" if use_abs_edge_color else edge_value_attr
            cbar_edges.set_label(label.capitalize())

        # Define o título do gráfico
        ax.set_title("Visualizacao do grafo")
        ax.axis("off")
        fig.tight_layout()
        plt.show()

    def plot_antigo(self, show_node_labels=True, show_edge_labels=True, precision=2, layout="planar"):
        G = self.get_network()

        if layout == "planar":
            try:
                pos = nx.planar_layout(G)
            except:
                pos = nx.spring_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(
                "layout must be one of: 'planar', 'spring', 'circular', 'shell', 'kamada_kawai', 'spectral'"
            )

        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

        if self.kind == "Directed":
            nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=20)
        else:
            nx.draw_networkx_edges(G, pos)

        if show_node_labels:
            node_labels = {
                node: node + "\n" + "\n".join([
                    f"{k[0].upper()}:{val:.{precision}f}" if isinstance(val, (int, float)) else f"{k[0].upper()}:{val}"
                    for k, val in attr.items()
                ])
                for node, attr in G.nodes(data=True)
            }

            nx.draw_networkx_labels(
                G,
                pos,
                labels=node_labels,
                font_color='black',
                font_size=9
            )

        if show_edge_labels:
            edge_labels = {
                (u, v): "\n".join([
                    f"{k[0].upper()}:{val:.{precision}f}" if isinstance(val, (int, float)) else f"{k[0].upper()}:{val}"
                    for k, val in data.items()
                ])
                for u, v, data in G.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_color='red',
                font_size=9
            )

        plt.title("Visualização do Grafo")
        plt.axis('off')
        plt.show()