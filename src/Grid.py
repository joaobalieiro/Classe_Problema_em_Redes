import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm, colors as mcolors
import networkx as nx
import numpy as np

from src.ProblemaP1 import ProblemaP1
from src.ProblemaP2 import ProblemaP2
import warnings
warnings.filterwarnings("ignore")

class GridP1(ProblemaP1):

    """
    Grid n (linhas) × m (colunas) de canos com propriedades uniformes.
    Arestas direcionadas: esquerda→direita e cima→baixo.
    Entrada: nó (0, 0)         — pressão alta (Q_ext > 0)
    Saída:   nó (n-1, m-1)     — pressão baixa (Q_ext < 0)
    """

    def __init__(self, n, m, area, length, mu, patm,
                 Q_in=1.0, Q_out=None):
        """
        n, m    : dimensões do grid (linhas × colunas)
        area    : área da seção transversal dos canos
        length  : comprimento de cada cano
        mu      : viscosidade
        patm    : pressão de referência (saída)
        Q_in    : fluxo de entrada concentrado no nó (0, 0)
        Q_out   : fluxo de saída concentrado no nó (n-1, m-1)
        """
        self.n = n
        self.m = m

        nodes, edges = self._build_grid(n, m, area, length)

        Q_ext = self._build_Q_ext(n, m, Q_in, Q_out)

        super().__init__(nodes=nodes, edges=edges, mu=mu, patm=patm, Q_ext=Q_ext)

    # ── construtores internos ─────────────────────────────────────────────────

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
                # horizontal: esquerda → direita
                if j + 1 < m:
                    v = self._node_id(i, j + 1)
                    edges[u][v] = dict(pipe)
                # vertical: cima → baixo
                if i + 1 < n:
                    v = self._node_id(i + 1, j)
                    edges[u][v] = dict(pipe)

        return nodes, edges

    def _build_Q_ext(self, n, m, Q_in, Q_out):
        if Q_out is None:
            Q_out = -Q_in

        Q_ext = {}
        # Zera o fluxo em todos os nós inicialmente
        for i in range(n):
            for j in range(m):
                nid = self._node_id(i, j)
                Q_ext[nid] = 0.0

        # Define pontualmente a entrada e a saída
        Q_ext[self._node_id(0, 0)] = Q_in
        Q_ext[self._node_id(n - 1, m - 1)] = Q_out

        return Q_ext

    # ── posições em grid para o plot ──────────────────────────────────────────

    def _grid_positions(self):
        pos = {}
        for i in range(self.n):
            for j in range(self.m):
                nid = self._node_id(i, j)
                pos[nid] = (j, -(i))   # x=coluna, y=-linha (cima→baixo)
        return pos

    # ── plot especializado ────────────────────────────────────────────────────

    def plot_grid(self, figsize=None, precision=2,
                  node_cmap="coolwarm", edge_cmap="plasma",
                  show_node_labels=True, show_edge_labels=False):

        G = self.get_network()
        pos = self._grid_positions()

        if figsize is None:
            figsize = (max(6, self.m * 1.8), max(5, self.n * 1.8))

        fig, ax = plt.subplots(figsize=figsize)

        node_sm = None
        edge_sm = None

        # ── cores dos nós (pressão) ───────────────────────────────────────────
        node_order = list(G.nodes())
        pressures = np.array([G.nodes[nd].get("pressao", np.nan) for nd in node_order], dtype=float)

        has_p = np.isfinite(pressures).any()
        if has_p:
            vmin, vmax = pressures[np.isfinite(pressures)].min(), pressures[np.isfinite(pressures)].max()
            if np.isclose(vmin, vmax): vmax = vmin + 1
            node_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            nx.draw_networkx_nodes(G, pos, node_size=700,
                                   node_color=pressures,
                                   cmap=cm.get_cmap(node_cmap),
                                   vmin=vmin, vmax=vmax, ax=ax)
            
            node_sm = cm.ScalarMappable(norm=node_norm, cmap=cm.get_cmap(node_cmap))
            node_sm.set_array([])
        else:
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", ax=ax)

        # ── cores das arestas (vazão) ───────────────────────────────────────
        edge_order = list(G.edges())
        flows = np.array([abs(G.edges[e].get("vazao", np.nan)) for e in edge_order], dtype=float)

        has_q = np.isfinite(flows).any()
        if has_q:
            fmin, fmax = flows[np.isfinite(flows)].min(), flows[np.isfinite(flows)].max()
            if np.isclose(fmin, fmax): fmax = fmin + 1
            edge_norm = mcolors.Normalize(vmin=fmin, vmax=fmax)
            widths = np.interp(flows, (fmin, fmax), (1.5, 5.5))
            nx.draw_networkx_edges(G, pos,
                                   edge_color=flows,
                                   edge_cmap=cm.get_cmap(edge_cmap),
                                   edge_vmin=fmin, edge_vmax=fmax,
                                   width=widths,
                                   arrows=True,
                                   arrowstyle="-|>", arrowsize=15,
                                   connectionstyle="arc3,rad=0.0",
                                   ax=ax)
            
            edge_sm = cm.ScalarMappable(norm=edge_norm, cmap=cm.get_cmap(edge_cmap))
            edge_sm.set_array([])
        else:
            nx.draw_networkx_edges(G, pos, width=2, edge_color="gray",
                                   arrows=True, arrowstyle="-|>", arrowsize=15, ax=ax)

        # ── labels dos nós ────────────────────────────────────────────────────
        if show_node_labels:
            labels = {}
            for nd in node_order:
                p_val = G.nodes[nd].get("pressao", None)
                label = nd
                if p_val is not None:
                    label += f"\n{p_val:.{precision}f}"
                labels[nd] = label
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        # ── labels das arestas ────────────────────────────────────────────────
        if show_edge_labels:
            elabels = {
                (u, v): f"{G.edges[u,v].get('vazao', 0):.{precision}f}"
                for u, v in edge_order
                if "vazao" in G.edges[u, v]
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels,
                                         font_size=6, rotate=False, ax=ax)

        # ── legenda de bordas ─────────────────────────────────────────────────
        patch_in  = mpatches.Patch(color="green",  alpha=0.4, label="Entrada (0, 0)")
        patch_out = mpatches.Patch(color="red",    alpha=0.4, label=f"Saída ({self.n-1}, {self.m-1})")
        ax.legend(handles=[patch_in, patch_out], loc="upper right", fontsize=8)
    
        # ── destaca nós de entrada e saída ────────────────────────────────────
        x_in,  y_in  = pos[self._node_id(0, 0)]
        x_out, y_out = pos[self._node_id(self.n - 1, self.m - 1)]
        
        ax.scatter(x_in,  y_in,  s=900, color="green", alpha=0.25, zorder=0)
        ax.scatter(x_out, y_out, s=900, color="red",   alpha=0.25, zorder=0)

        ax.set_title(f"Escoamento em Rede", fontsize=12)
        ax.axis("off")
        
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        bbox = ax.get_position()
        
        # -------------------------
        # Colorbar
        # -------------------------
        if has_p and node_sm is not None:
            cax_nodes = fig.add_axes([
                bbox.x1 + 0.02,               # x
                bbox.y0 + bbox.height * 0.52, # y
                0.03,                         # largura
                bbox.height * 0.40,           # altura
            ])
            cbar_nodes = fig.colorbar(node_sm, cax=cax_nodes)
            cbar_nodes.set_label("Pressão", fontsize=10)
            cbar_nodes.ax.tick_params(labelsize=9)

        if has_q and edge_sm is not None:
            cax_edges = fig.add_axes([
                bbox.x1 + 0.02,               # x
                bbox.y0 + 0.05,               # y
                0.03,                         # largura
                bbox.height * 0.40,           # altura
            ])
            cbar_edges = fig.colorbar(edge_sm, cax=cax_edges)
            cbar_edges.set_label("Vazão", fontsize=10)
            cbar_edges.ax.tick_params(labelsize=9)

        plt.show()


class GridP2(ProblemaP2):
    """
    Grid n (linhas) × m (colunas) para o Problema Probabilístico Direto (P2).
    
    Herda de ProblemaP2 para estimar a probabilidade de falha (pressão > P_max)
    via simulações de Monte Carlo com obstruções aleatórias nas arestas.
    
    A rede base (sem obstruções) tem as propriedades uniformes.
    Entrada: nó (0, 0)
    Saída:   nó (n-1, m-1)
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
        seed=None
    ):
        """
        Parâmetros da Rede (GridP1):
        ----------------------------
        n, m    : dimensões do grid (linhas × colunas)
        area    : área da seção transversal dos canos
        length  : comprimento de cada cano
        mu      : viscosidade
        patm    : pressão de referência (saída)
        Q_in    : fluxo de entrada concentrado no nó (0, 0)
        Q_out   : fluxo de saída concentrado no nó (n-1, m-1)
        
        Parâmetros Probabilísticos (ProblemaP2):
        ----------------------------------------
        r_prob     : probabilidade de obstrução de cada aresta
        alpha      : fator de penalização da condutância
        P_max      : pressão máxima admissível (limiar de falha)
        n_problems : número de instâncias independentes (experimentos MC)
        n_samples  : número de simulações por instância
        seed       : semente aleatória
        """
        self.n = n
        self.m = m

        # 1. Constrói a topologia base (igual ao GridP1)
        nodes, edges = self._build_grid(n, m, area, length)
        Q_ext = self._build_Q_ext(n, m, Q_in, Q_out)

        # 2. Inicializa o ProblemaP2 passando os dados da rede e os parâmetros estocásticos
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
            seed=seed
        )

    # ── construtores internos da rede ─────────────────────────────────────────

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
                # horizontal: esquerda → direita
                if j + 1 < m:
                    v = self._node_id(i, j + 1)
                    edges[u][v] = dict(pipe)
                # vertical: cima → baixo
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

    # ── posições em grid para o plot ──────────────────────────────────────────

    def _grid_positions(self):
        pos = {}
        for i in range(self.n):
            for j in range(self.m):
                nid = self._node_id(i, j)
                pos[nid] = (j, -(i))   # x=coluna, y=-linha (cima→baixo)
        return pos

    # ── plot especializado (exibe o cenário base) ─────────────────────────────

    def plot_grid(self, figsize=None, precision=2,
                  node_cmap="coolwarm", edge_cmap="plasma",
                  show_node_labels=True, show_edge_labels=False):
        """
        Plota o estado base da rede (sem considerar as obstruções aplicadas 
        durante a simulação de Monte Carlo). Caso queira visualizar as pressões
        base, certifique-se de chamar `.solve()` antes de plotar.
        """
        G = self.get_network()
        pos = self._grid_positions()

        if figsize is None:
            figsize = (max(6, self.m * 1.8), max(5, self.n * 1.8))

        fig, ax = plt.subplots(figsize=figsize)

        node_sm = None
        edge_sm = None

        # ── cores dos nós (pressão) ───────────────────────────────────────────
        node_order = list(G.nodes())
        pressures = np.array([G.nodes[nd].get("pressao", np.nan) for nd in node_order], dtype=float)

        has_p = np.isfinite(pressures).any()
        if has_p:
            vmin, vmax = pressures[np.isfinite(pressures)].min(), pressures[np.isfinite(pressures)].max()
            if np.isclose(vmin, vmax): vmax = vmin + 1
            node_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            nx.draw_networkx_nodes(G, pos, node_size=700,
                                   node_color=pressures,
                                   cmap=cm.get_cmap(node_cmap),
                                   vmin=vmin, vmax=vmax, ax=ax)
            
            node_sm = cm.ScalarMappable(norm=node_norm, cmap=cm.get_cmap(node_cmap))
            node_sm.set_array([])
        else:
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", ax=ax)

        # ── cores das arestas (vazão) ───────────────────────────────────────
        edge_order = list(G.edges())
        flows = np.array([abs(G.edges[e].get("vazao", np.nan)) for e in edge_order], dtype=float)

        has_q = np.isfinite(flows).any()
        if has_q:
            fmin, fmax = flows[np.isfinite(flows)].min(), flows[np.isfinite(flows)].max()
            if np.isclose(fmin, fmax): fmax = fmin + 1
            edge_norm = mcolors.Normalize(vmin=fmin, vmax=fmax)
            widths = np.interp(flows, (fmin, fmax), (1.5, 5.5))
            nx.draw_networkx_edges(G, pos,
                                   edge_color=flows,
                                   edge_cmap=cm.get_cmap(edge_cmap),
                                   edge_vmin=fmin, edge_vmax=fmax,
                                   width=widths,
                                   arrows=True,
                                   arrowstyle="-|>", arrowsize=15,
                                   connectionstyle="arc3,rad=0.0",
                                   ax=ax)
            
            edge_sm = cm.ScalarMappable(norm=edge_norm, cmap=cm.get_cmap(edge_cmap))
            edge_sm.set_array([])
        else:
            nx.draw_networkx_edges(G, pos, width=2, edge_color="gray",
                                   arrows=True, arrowstyle="-|>", arrowsize=15, ax=ax)

        # ── labels dos nós ────────────────────────────────────────────────────
        if show_node_labels:
            labels = {}
            for nd in node_order:
                p_val = G.nodes[nd].get("pressao", None)
                label = nd
                if p_val is not None:
                    label += f"\n{p_val:.{precision}f}"
                labels[nd] = label
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        # ── labels das arestas ────────────────────────────────────────────────
        if show_edge_labels:
            elabels = {
                (u, v): f"{G.edges[u,v].get('vazao', 0):.{precision}f}"
                for u, v in edge_order
                if "vazao" in G.edges[u, v]
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels,
                                         font_size=6, rotate=False, ax=ax)

        # ── legenda de bordas ─────────────────────────────────────────────────
        patch_in  = mpatches.Patch(color="green",  alpha=0.4, label="Entrada (0, 0)")
        patch_out = mpatches.Patch(color="red",    alpha=0.4, label=f"Saída ({self.n-1}, {self.m-1})")
        ax.legend(handles=[patch_in, patch_out], loc="upper right", fontsize=8)
    
        # ── destaca nós de entrada e saída ────────────────────────────────────
        x_in,  y_in  = pos[self._node_id(0, 0)]
        x_out, y_out = pos[self._node_id(self.n - 1, self.m - 1)]
        
        ax.scatter(x_in,  y_in,  s=900, color="green", alpha=0.25, zorder=0)
        ax.scatter(x_out, y_out, s=900, color="red",   alpha=0.25, zorder=0)

        ax.set_title("Escoamento Base em Rede (Cenário Nominal)", fontsize=12)
        ax.axis("off")
        
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        bbox = ax.get_position()
        
        # -------------------------
        # Colorbars
        # -------------------------
        if has_p and node_sm is not None:
            cax_nodes = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + bbox.height * 0.52, 0.03, bbox.height * 0.40])
            cbar_nodes = fig.colorbar(node_sm, cax=cax_nodes)
            cbar_nodes.set_label("Pressão", fontsize=10)
            cbar_nodes.ax.tick_params(labelsize=9)

        if has_q and edge_sm is not None:
            cax_edges = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + 0.05, 0.03, bbox.height * 0.40])
            cbar_edges = fig.colorbar(edge_sm, cax=cax_edges)
            cbar_edges.set_label("Vazão", fontsize=10)
            cbar_edges.ax.tick_params(labelsize=9)

        plt.show()