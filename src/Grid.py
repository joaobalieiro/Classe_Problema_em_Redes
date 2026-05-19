import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm, colors as mcolors
import networkx as nx
import numpy as np

from src.ProblemaP1 import ProblemaP1
import warnings
warnings.filterwarnings("ignore")

class GridP1(ProblemaP1):
    """
    Grid n (linhas) × m (colunas) de canos com propriedades uniformes.
    Arestas direcionadas: esquerda→direita e cima→baixo.
    """
    def __init__(self, n, m, area, length, mu, patm,
                 Q_in=1.0, Q_out=None, inflow_pressure=None, **kwargs):
        self.n = n
        self.m = m

        nodes, edges = self._build_grid(n, m, area, length)
        Q_ext = self._build_Q_ext(n, m, Q_in, Q_out)
        self.is_grid = True

        super().__init__(
            nodes=nodes,
            edges=edges,
            mu=mu,
            patm=patm,
            Q_ext=Q_ext,
            inflow_pressure=inflow_pressure,
            **kwargs
        )

    @staticmethod
    def _node_id(i, j):
        return f"n{i},{j}"

    def _build_grid(self, n, m, area, length):
        nodes, edges = {}, {}

        for i in range(n):
            for j in range(m):
                nid = self._node_id(i, j)
                nodes[nid], edges[nid] = {}, {}

        pipe = {"area": area, "length": length}

        for i in range(n):
            for j in range(m):
                u = self._node_id(i, j)

                if j + 1 < m:
                    edges[u][self._node_id(i, j + 1)] = dict(pipe)

                if i + 1 < n:
                    edges[u][self._node_id(i + 1, j)] = dict(pipe)

        return nodes, edges

    def _build_Q_ext(self, n, m, Q_in, Q_out):
        if Q_out is None:
            Q_out = -Q_in

        Q_ext = {
            self._node_id(i, j): 0.0
            for i in range(n)
            for j in range(m)
        }

        Q_ext[self._node_id(0, 0)] = Q_in
        Q_ext[self._node_id(n - 1, m - 1)] = Q_out

        return Q_ext

    def _grid_positions(self):
        return {
            self._node_id(i, j): (j, -i)
            for i in range(self.n)
            for j in range(self.m)
        }

    def plot_grid(
        self,
        figsize=None,
        precision=2,
        node_cmap="coolwarm",
        edge_cmap="plasma",
        show_node_labels=True,
        show_edge_labels=False
    ):
        G = self.get_network()
        pos = self._grid_positions()

        if figsize is None:
            figsize = (max(6, self.m * 1.8), max(5, self.n * 1.8))

        fig, ax = plt.subplots(figsize=figsize)
        node_sm, edge_sm = None, None

        # Nós (pressão)
        node_order = list(G.nodes())
        pressures = np.array(
            [G.nodes[nd].get("pressao", np.nan) for nd in node_order],
            dtype=float
        )

        has_p = np.isfinite(pressures).any()

        if has_p:
            vmin, vmax = (
                pressures[np.isfinite(pressures)].min(),
                pressures[np.isfinite(pressures)].max()
            )

            if np.isclose(vmin, vmax):
                vmax = vmin + 1

            node_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=700,
                node_color=pressures,
                cmap=cm.get_cmap(node_cmap),
                vmin=vmin,
                vmax=vmax,
                ax=ax
            )

            node_sm = cm.ScalarMappable(
                norm=node_norm,
                cmap=cm.get_cmap(node_cmap)
            )
            node_sm.set_array([])

        else:
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=700,
                node_color="skyblue",
                ax=ax
            )

        # Arestas (vazão)
        edge_order = list(G.edges())

        flows = np.array(
            [abs(G.edges[e].get("vazao", np.nan)) for e in edge_order],
            dtype=float
        )

        has_q = np.isfinite(flows).any()

        if has_q:
            fmin, fmax = (
                flows[np.isfinite(flows)].min(),
                flows[np.isfinite(flows)].max()
            )

            if np.isclose(fmin, fmax):
                fmax = fmin + 1

            edge_norm = mcolors.Normalize(vmin=fmin, vmax=fmax)

            widths = np.interp(flows, (fmin, fmax), (1.5, 5.5))

            nx.draw_networkx_edges(
                G,
                pos,
                edge_color=flows,
                edge_cmap=cm.get_cmap(edge_cmap),
                edge_vmin=fmin,
                edge_vmax=fmax,
                width=widths,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=15,
                ax=ax
            )

            edge_sm = cm.ScalarMappable(
                norm=edge_norm,
                cmap=cm.get_cmap(edge_cmap)
            )
            edge_sm.set_array([])

        else:
            nx.draw_networkx_edges(
                G,
                pos,
                width=2,
                edge_color="gray",
                arrows=True,
                arrowstyle="-|>",
                arrowsize=15,
                ax=ax
            )

        # Labels dos nós
        if show_node_labels:
            labels = {
                nd: (
                    f"{nd}\n{G.nodes[nd]['pressao']:.{precision}f}"
                    if G.nodes[nd].get("pressao") is not None
                    else nd
                )
                for nd in node_order
            }

            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=7,
                ax=ax
            )

        # Labels das arestas
        if show_edge_labels:
            elabels = {
                (u, v): f"{G.edges[u, v].get('vazao', 0):.{precision}f}"
                for u, v in edge_order
                if "vazao" in G.edges[u, v]
            }

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=elabels,
                font_size=6,
                rotate=False,
                ax=ax
            )

        # Destaques entrada/saída
        ax.legend(
            handles=[
                mpatches.Patch(
                    color="green",
                    alpha=0.4,
                    label="Entrada (0, 0)"
                ),
                mpatches.Patch(
                    color="red",
                    alpha=0.4,
                    label=f"Saída ({self.n-1}, {self.m-1})"
                )
            ],
            loc="upper right",
            fontsize=8
        )

        x_in, y_in = pos[self._node_id(0, 0)]
        x_out, y_out = pos[self._node_id(self.n - 1, self.m - 1)]

        ax.scatter(x_in, y_in, s=900, color="green", alpha=0.25, zorder=0)
        ax.scatter(x_out, y_out, s=900, color="red", alpha=0.25, zorder=0)

        ax.set_title("Escoamento em Rede", fontsize=12)
        ax.axis("off")

        fig.tight_layout()
        fig.subplots_adjust(right=0.85)

        bbox = ax.get_position()

        # Colorbars
        if has_p and node_sm is not None:
            cax_nodes = fig.add_axes([
                bbox.x1 + 0.02,
                bbox.y0 + bbox.height * 0.52,
                0.03,
                bbox.height * 0.40
            ])

            fig.colorbar(node_sm, cax=cax_nodes).set_label(
                "Pressão",
                fontsize=10
            )

        if has_q and edge_sm is not None:
            cax_edges = fig.add_axes([
                bbox.x1 + 0.02,
                bbox.y0 + 0.05,
                0.03,
                bbox.height * 0.40
            ])

            fig.colorbar(edge_sm, cax=cax_edges).set_label(
                "Vazão",
                fontsize=10
            )

        plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm, colors as mcolors
import networkx as nx
import numpy as np

from src.ProblemaP2 import ProblemaP2


class GridP2(ProblemaP2):
    """
    Grid n (linhas) × m (colunas) para o Problema Probabilístico Direto (P2).

    Herda de ProblemaP2 para estimar a probabilidade de falha
    (pressão > P_max) via simulações de Monte Carlo com
    obstruções aleatórias nas arestas.
    """

    def __init__(
        self,
        n: int,
        m: int,
        area: float,
        length: float,
        mu: float,
        patm: float,
        Q_in: float = 1.0,
        Q_out: float = None,
        r_prob: float = 0.1,
        alpha: float = 2.0,
        P_max: float = 1e5,
        n_problems: int = 10,
        n_samples: int = 100,
        seed=None,
        inflow_pressure=None,
        **kwargs
    ):
        self.n = n
        self.m = m

        # Topologia da rede
        nodes, edges = self._build_grid(n, m, area, length)
        Q_ext = self._build_Q_ext(n, m, Q_in, Q_out)

        # Inicializa ProblemaP2
        super().__init__(
            nodes=nodes,
            edges=edges,
            mu=mu,
            patm=patm,
            Q_ext=Q_ext,
            r_prob=r_prob,
            alpha=alpha,
            P_max=P_max,
            n_problems=n_problems,
            n_samples=n_samples,
            seed=seed,
            inflow_pressure=inflow_pressure,
            **kwargs
        )

    @staticmethod
    def _node_id(i, j):
        return f"n{i},{j}"

    def _build_grid(self, n, m, area, length):
        nodes = {}
        edges = {}

        for i in range(n):
            for j in range(m):
                nid = self._node_id(i, j)
                nodes[nid] = {}
                edges[nid] = {}

        pipe = {"area": area, "length": length}

        for i in range(n):
            for j in range(m):
                u = self._node_id(i, j)

                # Horizontal
                if j + 1 < m:
                    v = self._node_id(i, j + 1)
                    edges[u][v] = dict(pipe)

                # Vertical
                if i + 1 < n:
                    v = self._node_id(i + 1, j)
                    edges[u][v] = dict(pipe)

        return nodes, edges

    def _build_Q_ext(self, n, m, Q_in, Q_out):
        if Q_out is None:
            Q_out = -Q_in

        Q_ext = {}

        for i in range(n):
            for j in range(m):
                Q_ext[self._node_id(i, j)] = 0.0

        Q_ext[self._node_id(0, 0)] = Q_in
        Q_ext[self._node_id(n - 1, m - 1)] = Q_out

        return Q_ext

    def _grid_positions(self):
        pos = {}

        for i in range(self.n):
            for j in range(self.m):
                nid = self._node_id(i, j)
                pos[nid] = (j, -i)

        return pos

    def plot_grid(
        self,
        figsize=None,
        precision=2,
        node_cmap="coolwarm",
        edge_cmap="plasma",
        show_node_labels=True,
        show_edge_labels=False
    ):
        """
        Plota o estado base nominal da rede.
        Certifique-se de invocar `.solve()` antes de plotar.
        """
        G = self.get_network()
        pos = self._grid_positions()

        if figsize is None:
            figsize = (max(6, self.m * 1.8), max(5, self.n * 1.8))

        fig, ax = plt.subplots(figsize=figsize)
        node_sm, edge_sm = None, None

        # Nós (pressão)
        node_order = list(G.nodes())

        pressures = np.array(
            [G.nodes[nd].get("pressao", np.nan) for nd in node_order],
            dtype=float
        )

        has_p = np.isfinite(pressures).any()

        if has_p:
            vmin, vmax = (
                pressures[np.isfinite(pressures)].min(),
                pressures[np.isfinite(pressures)].max()
            )

            if np.isclose(vmin, vmax):
                vmax = vmin + 1

            node_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=700,
                node_color=pressures,
                cmap=cm.get_cmap(node_cmap),
                vmin=vmin,
                vmax=vmax,
                ax=ax
            )

            node_sm = cm.ScalarMappable(
                norm=node_norm,
                cmap=cm.get_cmap(node_cmap)
            )
            node_sm.set_array([])

        else:
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=700,
                node_color="skyblue",
                ax=ax
            )

        # Arestas (vazão)
        edge_order = list(G.edges())

        flows = np.array(
            [abs(G.edges[e].get("vazao", np.nan)) for e in edge_order],
            dtype=float
        )

        has_q = np.isfinite(flows).any()

        if has_q:
            fmin, fmax = (
                flows[np.isfinite(flows)].min(),
                flows[np.isfinite(flows)].max()
            )

            if np.isclose(fmin, fmax):
                fmax = fmin + 1

            edge_norm = mcolors.Normalize(vmin=fmin, vmax=fmax)

            widths = np.interp(flows, (fmin, fmax), (1.5, 5.5))

            nx.draw_networkx_edges(
                G,
                pos,
                edge_color=flows,
                edge_cmap=cm.get_cmap(edge_cmap),
                edge_vmin=fmin,
                edge_vmax=fmax,
                width=widths,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=15,
                ax=ax
            )

            edge_sm = cm.ScalarMappable(
                norm=edge_norm,
                cmap=cm.get_cmap(edge_cmap)
            )
            edge_sm.set_array([])

        else:
            nx.draw_networkx_edges(
                G,
                pos,
                width=2,
                edge_color="gray",
                arrows=True,
                arrowstyle="-|>",
                arrowsize=15,
                ax=ax
            )

        # Labels dos nós
        if show_node_labels:
            labels = {
                nd: (
                    f"{nd}\n{G.nodes[nd]['pressao']:.{precision}f}"
                    if G.nodes[nd].get("pressao") is not None
                    else nd
                )
                for nd in node_order
            }

            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=7,
                ax=ax
            )

        # Labels das arestas
        if show_edge_labels:
            elabels = {
                (u, v): f"{G.edges[u, v].get('vazao', 0):.{precision}f}"
                for u, v in edge_order
                if "vazao" in G.edges[u, v]
            }

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=elabels,
                font_size=6,
                rotate=False,
                ax=ax
            )

        # Destaques entrada/saída
        ax.legend(
            handles=[
                mpatches.Patch(
                    color="green",
                    alpha=0.4,
                    label="Entrada (0, 0)"
                ),
                mpatches.Patch(
                    color="red",
                    alpha=0.4,
                    label=f"Saída ({self.n-1}, {self.m-1})"
                )
            ],
            loc="upper right",
            fontsize=8
        )

        x_in, y_in = pos[self._node_id(0, 0)]
        x_out, y_out = pos[self._node_id(self.n - 1, self.m - 1)]

        ax.scatter(x_in, y_in, s=900, color="green", alpha=0.25, zorder=0)
        ax.scatter(x_out, y_out, s=900, color="red", alpha=0.25, zorder=0)

        ax.set_title(
            "Escoamento Base em Rede (Cenário Nominal)",
            fontsize=12
        )

        ax.axis("off")

        fig.tight_layout()
        fig.subplots_adjust(right=0.85)

        bbox = ax.get_position()

        # Colorbars
        if has_p and node_sm is not None:
            cax_nodes = fig.add_axes([
                bbox.x1 + 0.02,
                bbox.y0 + bbox.height * 0.52,
                0.03,
                bbox.height * 0.40
            ])

            fig.colorbar(node_sm, cax=cax_nodes).set_label(
                "Pressão",
                fontsize=10
            )

        if has_q and edge_sm is not None:
            cax_edges = fig.add_axes([
                bbox.x1 + 0.02,
                bbox.y0 + 0.05,
                0.03,
                bbox.height * 0.40
            ])

            fig.colorbar(edge_sm, cax=cax_edges).set_label(
                "Vazão",
                fontsize=10
            )

        plt.show()