import copy
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from src.ProblemaP1 import ProblemaP1


class ProblemaP4(ProblemaP1):
    """
    problema de otimizacao mista para distribuicao de canos grossos

    a classe recebe uma instancia do problema p1 e escolhe em quais arestas
    devem ser colocados os canos grossos para minimizar a maior pressao
    da rede, assumindo que a menor pressao e a pressao de saida prescrita
    """

    def __init__(
        self,
        *args,
        p1_instance=None,
        max_thick_pipes=1,
        thick_area_factor=2.0,
        use_exact_number=True,
        fixed_nodes=None,
        fixed_pressure=None,
        pressure_bounds=None,
        pressure_bound_factor=5.0,
        big_m_factor=1.05,
        solver_options=None,
        **kwargs,
    ):
        if p1_instance is not None:
            if not isinstance(p1_instance, ProblemaP1):
                raise TypeError("p1_instance deve ser uma instancia de ProblemaP1")
            self.__dict__.update(copy.deepcopy(p1_instance).__dict__)
        else:
            super().__init__(*args, **kwargs)

        self.max_thick_pipes = int(max_thick_pipes)
        self.thick_area_factor = float(thick_area_factor)
        self.use_exact_number = bool(use_exact_number)
        self.fixed_nodes = None if fixed_nodes is None else list(fixed_nodes)
        self.fixed_pressure = self.patm if fixed_pressure is None else float(fixed_pressure)
        self.pressure_bounds = pressure_bounds
        self.pressure_bound_factor = float(pressure_bound_factor)
        self.big_m_factor = float(big_m_factor)
        self.solver_options = solver_options or {"disp": False}

        self.p4_result = None
        self.x_solution = None
        self.selected_thick_edges = None
        self.objective_value = None
        self.fine_conductivities = None
        self.thick_conductivities = None

        self._validate_p4_parameters()

    def _validate_p4_parameters(self):
        if self.max_thick_pipes < 0:
            raise ValueError("max_thick_pipes deve ser nao negativo")
        if self.thick_area_factor <= 1.0:
            raise ValueError("thick_area_factor deve ser maior que 1")
        if self.pressure_bound_factor <= 0:
            raise ValueError("pressure_bound_factor deve ser positivo")
        if self.big_m_factor < 1.0:
            raise ValueError("big_m_factor deve ser maior ou igual a 1")

    def _ensure_ready(self):
        if self.Q_ext is None:
            raise ValueError("Q_ext deve estar definido")
        if self.A is None or self.K is None or self.M is None or not self.is_fitted:
            self.setup()
        if self.max_thick_pipes > self.num_edges:
            raise ValueError("max_thick_pipes nao pode ser maior que o numero de arestas")

    def _get_fixed_indices(self):
        node_order = self.get_node_order()
        node_to_index = {node: idx for idx, node in enumerate(node_order)}

        if self.fixed_nodes is not None:
            missing = [node for node in self.fixed_nodes if node not in node_to_index]
            if missing:
                raise ValueError(f"fixed_nodes contem nos desconhecidos: {missing}")
            return np.array([node_to_index[node] for node in self.fixed_nodes], dtype=int)

        return np.array(
            [
                idx
                for idx, node in enumerate(node_order)
                if float(self.Q_ext[node]) < 0.0
            ],
            dtype=int,
        )

    def _compute_fine_and_thick_conductivities(self):
        fine = []
        thick = []

        for u, v in self.get_edge_list():
            attrs = self.edges[u][v]
            area = float(attrs["area"])
            length = float(attrs["length"])

            c_fine = self._compute_conductivity(area, length)
            c_thick = self._compute_conductivity(area * self.thick_area_factor, length)

            fine.append(c_fine)
            thick.append(c_thick)

        self.fine_conductivities = np.array(fine, dtype=float)
        self.thick_conductivities = np.array(thick, dtype=float)
        return self.fine_conductivities, self.thick_conductivities

    def _solve_linear_with_conductivities(self, conductivities):
        self._ensure_ready()

        conductivities = np.asarray(conductivities, dtype=float)
        if conductivities.size != self.num_edges:
            raise ValueError("conductivities deve ter uma entrada por aresta")

        node_order = self.get_node_order()
        b = np.array([self.Q_ext[node] for node in node_order], dtype=float)
        fixed_indices = self._get_fixed_indices()

        if fixed_indices.size == 0:
            raise ValueError("nenhum no de pressao fixa foi identificado")

        n_nodes = len(node_order)
        all_indices = np.arange(n_nodes, dtype=int)
        free_indices = np.setdiff1d(all_indices, fixed_indices)

        K_mat = diags(conductivities, offsets=0, format="csr", dtype=float)
        M_mat = (self.A.T @ K_mat @ self.A).tocsr()

        p = np.zeros(n_nodes, dtype=float)
        p[fixed_indices] = self.fixed_pressure

        if free_indices.size > 0:
            free_free_matrix = M_mat[free_indices][:, free_indices]
            free_fixed_matrix = M_mat[free_indices][:, fixed_indices]
            free_rhs = b[free_indices] - free_fixed_matrix @ p[fixed_indices]
            p[free_indices] = np.asarray(spsolve(free_free_matrix, free_rhs), dtype=float)

        q = np.asarray(K_mat @ (self.A @ p), dtype=float).reshape(-1)
        return p, q

    def _infer_pressure_bounds(self):
        if self.pressure_bounds is not None:
            lower, upper = self.pressure_bounds
            if lower >= upper:
                raise ValueError("pressure_bounds deve satisfazer lower < upper")
            return float(lower), float(upper)

        fine, thick = self._compute_fine_and_thick_conductivities()
        p_fine, _ = self._solve_linear_with_conductivities(fine)
        p_thick, _ = self._solve_linear_with_conductivities(thick)

        values = np.concatenate([p_fine, p_thick, np.array([self.fixed_pressure])])
        p_min = float(np.min(values))
        p_max = float(np.max(values))
        span = p_max - p_min

        if np.isclose(span, 0.0):
            span = max(1.0, abs(p_max), abs(p_min))

        margin = self.pressure_bound_factor * span
        return p_min - margin, p_max + margin

    def _variable_indices(self):
        n_nodes = self.num_nodes
        n_edges = self.num_edges

        return {
            "p": np.arange(0, n_nodes, dtype=int),
            "q": np.arange(n_nodes, n_nodes + n_edges, dtype=int),
            "x": np.arange(n_nodes + n_edges, n_nodes + 2 * n_edges, dtype=int),
            "p_max": n_nodes + 2 * n_edges,
            "n_var": n_nodes + 2 * n_edges + 1,
        }

    def _add_constraint(self, rows, cols, data, lower, upper, row_data, lb, ub):
        row = len(lower)
        for col, value in row_data.items():
            if abs(value) > 0.0:
                rows.append(row)
                cols.append(int(col))
                data.append(float(value))
        lower.append(float(lb) if np.isfinite(lb) else -np.inf)
        upper.append(float(ub) if np.isfinite(ub) else np.inf)

    def _build_milp(self):
        self._ensure_ready()

        node_order = self.get_node_order()
        edge_list = self.get_edge_list()
        node_index = {node: idx for idx, node in enumerate(node_order)}

        n_nodes = self.num_nodes
        n_edges = self.num_edges
        idx = self._variable_indices()

        fine, thick = self._compute_fine_and_thick_conductivities()
        p_lower, p_upper = self._infer_pressure_bounds()
        pressure_span = p_upper - p_lower

        rows = []
        cols = []
        data = []
        lower = []
        upper = []

        b = np.array([self.Q_ext[node] for node in node_order], dtype=float)
        fixed_indices = self._get_fixed_indices()
        free_indices = np.setdiff1d(np.arange(n_nodes, dtype=int), fixed_indices)
        A = self.A.tocsr()

        for i in free_indices:
            row_data = {}
            for e in A[:, i].nonzero()[0]:
                row_data[idx["q"][e]] = float(A[e, i])
            self._add_constraint(rows, cols, data, lower, upper, row_data, b[i], b[i])

        for i in fixed_indices:
            row_data = {idx["p"][i]: 1.0}
            self._add_constraint(rows, cols, data, lower, upper, row_data, self.fixed_pressure, self.fixed_pressure)

        row_data = {int(col): 1.0 for col in idx["x"]}
        if self.use_exact_number:
            self._add_constraint(rows, cols, data, lower, upper, row_data, self.max_thick_pipes, self.max_thick_pipes)
        else:
            self._add_constraint(rows, cols, data, lower, upper, row_data, 0.0, self.max_thick_pipes)

        for i in range(n_nodes):
            self._add_constraint(
                rows,
                cols,
                data,
                lower,
                upper,
                {idx["p"][i]: 1.0, idx["p_max"]: -1.0},
                -np.inf,
                0.0,
            )

        for e, (u, v) in enumerate(edge_list):
            i = node_index[u]
            j = node_index[v]
            c_f = fine[e]
            c_g = thick[e]
            big_m = self.big_m_factor * max(abs(c_f), abs(c_g)) * pressure_span

            self._add_constraint(
                rows,
                cols,
                data,
                lower,
                upper,
                {
                    idx["q"][e]: 1.0,
                    idx["p"][i]: -c_f,
                    idx["p"][j]: c_f,
                    idx["x"][e]: -big_m,
                },
                -np.inf,
                0.0,
            )
            self._add_constraint(
                rows,
                cols,
                data,
                lower,
                upper,
                {
                    idx["q"][e]: 1.0,
                    idx["p"][i]: -c_f,
                    idx["p"][j]: c_f,
                    idx["x"][e]: big_m,
                },
                0.0,
                np.inf,
            )
            self._add_constraint(
                rows,
                cols,
                data,
                lower,
                upper,
                {
                    idx["q"][e]: 1.0,
                    idx["p"][i]: -c_g,
                    idx["p"][j]: c_g,
                    idx["x"][e]: big_m,
                },
                -np.inf,
                big_m,
            )
            self._add_constraint(
                rows,
                cols,
                data,
                lower,
                upper,
                {
                    idx["q"][e]: 1.0,
                    idx["p"][i]: -c_g,
                    idx["p"][j]: c_g,
                    idx["x"][e]: -big_m,
                },
                -big_m,
                np.inf,
            )

        A_ub = coo_matrix((data, (rows, cols)), shape=(len(lower), idx["n_var"])).tocsr()
        constraints = LinearConstraint(A_ub, np.array(lower), np.array(upper))

        c_obj = np.zeros(idx["n_var"], dtype=float)
        c_obj[idx["p_max"]] = 1.0

        lb = np.full(idx["n_var"], -np.inf, dtype=float)
        ub = np.full(idx["n_var"], np.inf, dtype=float)

        lb[idx["p"]] = p_lower
        ub[idx["p"]] = p_upper
        lb[idx["q"]] = -np.inf
        ub[idx["q"]] = np.inf
        lb[idx["x"]] = 0.0
        ub[idx["x"]] = 1.0
        lb[idx["p_max"]] = p_lower
        ub[idx["p_max"]] = p_upper

        bounds = Bounds(lb, ub)

        integrality = np.zeros(idx["n_var"], dtype=int)
        integrality[idx["x"]] = 1

        metadata = {
            "idx": idx,
            "node_order": node_order,
            "edge_list": edge_list,
            "pressure_bounds": (p_lower, p_upper),
            "fine_conductivities": fine,
            "thick_conductivities": thick,
        }

        return c_obj, integrality, bounds, constraints, metadata

    def solve_milp(self, accept_feasible=True, use_fallback=True):
        c_obj, integrality, bounds, constraints, metadata = self._build_milp()

        result = milp(
            c=c_obj,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=self.solver_options,
        )

        self.p4_result = result

        if result.success:
            self.solution_status = "otima"
            self.solution_is_optimal = True
            self._load_solution(result.x, metadata)
            return self.summary()

        if accept_feasible and result.x is not None:
            idx = metadata["idx"]
            x_raw = np.asarray(result.x[idx["x"]], dtype=float)
            x_round = np.rint(x_raw)
            is_integral = np.allclose(x_raw, x_round, atol=1e-5)

            if self.use_exact_number:
                count_ok = int(np.sum(x_round)) == int(self.max_thick_pipes)
            else:
                count_ok = int(np.sum(x_round)) <= int(self.max_thick_pipes)

            if is_integral and count_ok:
                self.solution_status = "viavel_nao_certificada"
                self.solution_is_optimal = False
                self._load_solution(result.x, metadata)
                return self.summary()

        if use_fallback:
            self.solution_status = "heuristica"
            self.solution_is_optimal = False
            candidate = self._heuristic_pressure_drop_solution()
            self._load_candidate_solution(candidate)
            return self.summary()

        raise RuntimeError(f"o solver nao retornou solucao utilizavel: {result.message}")

    def solve(self):
        return self.solve_milp()

    def _load_solution(self, solution, metadata):
        idx = metadata["idx"]
        node_order = metadata["node_order"]
        edge_list = metadata["edge_list"]
        fine = metadata["fine_conductivities"]
        thick = metadata["thick_conductivities"]

        p = np.asarray(solution[idx["p"]], dtype=float)
        x_raw = np.asarray(solution[idx["x"]], dtype=float)
        x = np.rint(x_raw).astype(int)
        conductivities = fine + (thick - fine) * x

        self.x_solution = x
        self.selected_thick_edges = [edge for edge, value in zip(edge_list, x) if value == 1]
        self.objective_value = float(solution[idx["p_max"]])

        self.K = diags(conductivities, offsets=0, format="csr", dtype=float)
        self.M = (self.A.T @ self.K @ self.A).tocsr()
        self.p = p
        self._update_node_pressures()
        self._compute_edge_flows()

        for e, (u, v) in enumerate(edge_list):
            attrs = self.edges[u][v]
            attrs["x_grosso"] = int(x[e])
            attrs["tipo_cano"] = "grosso" if x[e] == 1 else "fino"
            attrs["area_original"] = float(attrs["area"])
            attrs["area_usada"] = float(attrs["area"] * (self.thick_area_factor if x[e] == 1 else 1.0))
            attrs["condutancia_usada"] = float(conductivities[e])

    def evaluate_configuration(self, thick_edges=None):
        self._ensure_ready()

        if self.fine_conductivities is None or self.thick_conductivities is None:
            self._compute_fine_and_thick_conductivities()

        edge_list = self.get_edge_list()
        thick_set = set(thick_edges or [])
        x = np.array([1 if edge in thick_set else 0 for edge in edge_list], dtype=int)
        conductivities = self.fine_conductivities + (self.thick_conductivities - self.fine_conductivities) * x
        p, q = self._solve_linear_with_conductivities(conductivities)

        return {
            "thick_edges": [edge for edge, value in zip(edge_list, x) if value == 1],
            "pressure_min": float(np.min(p)),
            "pressure_max": float(np.max(p)),
            "pressure_range": float(np.max(p) - np.min(p)),
            "pressures": p,
            "flows": q,
        }

    def brute_force_check(self, max_combinations=200000):
        self._ensure_ready()
        edge_list = self.get_edge_list()
        combinations_count = 0
        best = None

        for comb in itertools.combinations(edge_list, self.max_thick_pipes):
            combinations_count += 1
            if combinations_count > max_combinations:
                raise RuntimeError("numero de combinacoes excedeu max_combinations")

            candidate = self.evaluate_configuration(thick_edges=comb)
            if best is None or candidate["pressure_max"] < best["pressure_max"]:
                best = candidate

        return best

    def summary(self):
        if self.p is None or self.x_solution is None:
            raise ValueError("execute solve_milp antes de chamar summary")

        return {
            "objective_pressure_max": float(np.max(self.p)),
            "solver_objective": float(self.objective_value),
            "pressure_min": float(np.min(self.p)),
            "pressure_max": float(np.max(self.p)),
            "pressure_range": float(np.max(self.p) - np.min(self.p)),
            "objective_pressure_range": float(np.max(self.p) - np.min(self.p)),
            "max_thick_pipes": int(self.max_thick_pipes),
            "n_thick_used": int(np.sum(self.x_solution)),
            "selected_thick_edges": list(self.selected_thick_edges),
            "solution_status": getattr(self, "solution_status", None),
            "solution_is_optimal": getattr(self, "solution_is_optimal", None),
            "solver_message": getattr(self.p4_result, "message", None) if self.p4_result is not None else None,
            "mip_gap": getattr(self.p4_result, "mip_gap", None) if self.p4_result is not None else None,
        }

    def edge_solution_table(self):
        if self.x_solution is None:
            raise ValueError("execute solve_milp antes de chamar edge_solution_table")

        rows = []
        for e, (u, v) in enumerate(self.get_edge_list()):
            attrs = self.edges[u][v]
            rows.append(
                {
                    "edge": (u, v),
                    "tipo_cano": attrs.get("tipo_cano"),
                    "x_grosso": int(attrs.get("x_grosso", 0)),
                    "area_original": float(attrs.get("area_original", attrs.get("area"))),
                    "area_usada": float(attrs.get("area_usada", attrs.get("area"))),
                    "length": float(attrs.get("length")),
                    "condutancia_usada": float(attrs.get("condutancia_usada")),
                    "delta_pressao": float(attrs.get("delta_pressao")),
                    "vazao": float(attrs.get("vazao")),
                }
            )

        return rows

    def node_solution_table(self):
        if self.p is None:
            raise ValueError("execute solve_milp antes de chamar node_solution_table")

        rows = []
        for node in self.get_node_order():
            rows.append(
                {
                    "node": node,
                    "pressao": float(self.nodes[node].get("pressao")),
                    "fluxo_externo": float(self.nodes[node].get("fluxo_externo", self.Q_ext[node])),
                }
            )

        return rows

    def plot_solution(
        self,
        layout="planar",
        figsize=(10, 8),
        node_cmap="viridis",
        edge_cmap="plasma",
        show_labels=True,
        seed=42,
    ):
        if self.p is None or self.x_solution is None:
            raise ValueError("execute solve_milp antes de chamar plot_solution")

        G = self.get_network()
        pos = self._get_layout_positions(G, layout=layout, seed=seed)
        fig, ax = plt.subplots(figsize=figsize)

        node_order = list(G.nodes())
        edge_order = list(G.edges())

        node_values = np.array([G.nodes[node].get("pressao", np.nan) for node in node_order], dtype=float)
        edge_values = np.array([abs(G.edges[edge].get("vazao", np.nan)) for edge in edge_order], dtype=float)
        edge_widths = []

        if np.isfinite(edge_values).any() and not np.isclose(np.nanmin(edge_values), np.nanmax(edge_values)):
            edge_widths = np.interp(edge_values, (np.nanmin(edge_values), np.nanmax(edge_values)), (1.5, 5.5))
        else:
            edge_widths = np.full(len(edge_order), 2.5)

        edge_styles = ["solid" if G.edges[edge].get("tipo_cano") == "grosso" else "dashed" for edge in edge_order]

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_values,
            cmap=plt.get_cmap(node_cmap),
            node_size=900,
            ax=ax,
        )

        for edge, width, style in zip(edge_order, edge_widths, edge_styles):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[edge],
                width=float(width),
                style=style,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=20,
                ax=ax,
            )

        if show_labels:
            labels = {node: f"{node}\np={G.nodes[node].get('pressao', np.nan):.3g}" for node in node_order}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

            edge_labels = {
                edge: f"{G.edges[edge].get('tipo_cano')}\nq={G.edges[edge].get('vazao', np.nan):.3g}"
                for edge in edge_order
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(node_cmap))
        sm.set_array(node_values)
        fig.colorbar(sm, ax=ax, label="pressao")

        ax.set_title("solucao do problema p4")
        ax.axis("off")
        fig.tight_layout()
        return fig, ax
    
    def _heuristic_pressure_drop_solution(self):
        self._ensure_ready()

        if self.fine_conductivities is None or self.thick_conductivities is None:
            self._compute_fine_and_thick_conductivities()

        edge_list = self.get_edge_list()

        p_fine, _ = self._solve_linear_with_conductivities(self.fine_conductivities)
        pressure_drops = np.abs(self.A @ p_fine)

        n_select = min(self.max_thick_pipes, self.num_edges)
        selected_indices = np.argsort(-pressure_drops)[:n_select]
        selected_edges = [edge_list[i] for i in selected_indices]

        return self.evaluate_configuration(thick_edges=selected_edges)


    def _load_candidate_solution(self, candidate):
        if self.fine_conductivities is None or self.thick_conductivities is None:
            self._compute_fine_and_thick_conductivities()

        edge_list = self.get_edge_list()
        thick_set = set(candidate["thick_edges"])

        x = np.array([1 if edge in thick_set else 0 for edge in edge_list], dtype=int)
        conductivities = self.fine_conductivities + (self.thick_conductivities - self.fine_conductivities) * x

        self.x_solution = x
        self.selected_thick_edges = list(candidate["thick_edges"])
        self.objective_value = float(candidate["pressure_max"])

        self.K = diags(conductivities, offsets=0, format="csr", dtype=float)
        self.M = (self.A.T @ self.K @ self.A).tocsr()
        self.p = np.asarray(candidate["pressures"], dtype=float)

        self._update_node_pressures()
        self._compute_edge_flows()

        for e, (u, v) in enumerate(edge_list):
            attrs = self.edges[u][v]
            attrs["x_grosso"] = int(x[e])
            attrs["tipo_cano"] = "grosso" if x[e] == 1 else "fino"
            attrs["area_original"] = float(attrs["area"])
            attrs["area_usada"] = float(attrs["area"] * (self.thick_area_factor if x[e] == 1 else 1.0))
            attrs["condutancia_usada"] = float(conductivities[e])