import numpy as np
from scipy.stats import norm
from scipy.sparse.linalg import spsolve
from src.ProblemaP2 import ProblemaP2

class ProblemaP3(ProblemaP2):
    """
    problema probabilistico inverso

    esta classe reaproveita a estrutura do problema P2 e testa os valores de r
    associados as frequencias de limpeza da tabela do problema P3

    nesta versao a pressao e fixada nos nos de entrada
    """

    DEFAULT_MAINTENANCE_TABLE = [
        {"frequencia": 1, "r_prob": 0.25, "custo": 5.0},
        {"frequencia": 2, "r_prob": 0.10, "custo": 10.0},
        {"frequencia": 3, "r_prob": 0.06, "custo": 15.0},
        {"frequencia": 4, "r_prob": 0.04, "custo": 20.0},
        {"frequencia": 6, "r_prob": 0.03, "custo": 30.0},
        {"frequencia": 8, "r_prob": 0.02, "custo": 40.0},
        {"frequencia": 12, "r_prob": 0.01, "custo": 60.0},
    ]

    def __init__(
        self,
        *args,
        p1_instance=None,
        alpha=2.0,
        P_in=1.0,
        P_min=0.0,
        x=0.05,
        n_samples=1000,
        seed=None,
        maintenance_table=None,
        monitor_nodes=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            p1_instance=p1_instance,
            r_prob=0.0,
            alpha=alpha,
            P_max=np.inf,
            n_samples=n_samples,
            seed=seed,
            **kwargs,
        )

        self.P_in = float(P_in)
        self.P_min = float(P_min)
        self.x = self._normalize_probability(x)
        self.maintenance_table = self._validate_maintenance_table(
            maintenance_table or self.DEFAULT_MAINTENANCE_TABLE
        )
        self.monitor_nodes = monitor_nodes
        self.p3_results = None
        self.best_solution = None

    @staticmethod
    def _normalize_probability(x):
        x = float(x)

        if x < 0:
            raise ValueError("x must be non negative")

        if x > 1:
            x = x / 100.0

        if x > 1:
            raise ValueError("x must be in [0, 1] or in [0, 100]")

        return x

    @staticmethod
    def _validate_maintenance_table(table):
        validated = []

        for row in table:
            if "frequencia" not in row or "r_prob" not in row or "custo" not in row:
                raise ValueError("each row must contain frequencia r_prob and custo")

            frequencia = int(row["frequencia"])
            r_prob = float(row["r_prob"])
            custo = float(row["custo"])

            if frequencia <= 0:
                raise ValueError("frequencia must be positive")
            if not 0 <= r_prob <= 1:
                raise ValueError("r_prob must be in [0, 1]")
            if custo < 0:
                raise ValueError("custo must be non negative")

            validated.append(
                {
                    "frequencia": frequencia,
                    "r_prob": r_prob,
                    "custo": custo,
                }
            )

        return sorted(validated, key=lambda row: row["frequencia"])

    def _ensure_ready(self):
        if self.Q_ext is None:
            raise ValueError("Q_ext must be defined")

        if self.A is None or self.K is None or self.M is None or not self.is_fitted:
            raise ValueError("call setup before running p3")

    def _get_input_indices(self):
        node_order = self.get_node_order()

        return np.array(
            [
                idx
                for idx, node in enumerate(node_order)
                if float(self.Q_ext[node]) > 0.0
            ],
            dtype=int,
        )

    def _get_monitor_indices(self):
        node_order = self.get_node_order()
        node_to_index = {node: idx for idx, node in enumerate(node_order)}

        if self.monitor_nodes is not None:
            missing = [node for node in self.monitor_nodes if node not in node_to_index]

            if missing:
                raise ValueError(f"monitor_nodes contains unknown nodes: {missing}")

            return np.array([node_to_index[node] for node in self.monitor_nodes], dtype=int)

        input_indices = set(self._get_input_indices())

        return np.array(
            [
                idx
                for idx, _ in enumerate(node_order)
                if idx not in input_indices
            ],
            dtype=int,
        )

    def _solve_with_input_pressure(self, problem):
        node_order = problem.get_node_order()
        b = np.array([problem.Q_ext[node] for node in node_order], dtype=float)

        fixed_indices = self._get_input_indices()

        if fixed_indices.size == 0:
            raise ValueError("no input node with positive flow found")

        n_nodes = len(node_order)
        all_indices = np.arange(n_nodes, dtype=int)
        free_indices = np.setdiff1d(all_indices, fixed_indices)

        p = np.zeros(n_nodes, dtype=float)
        p[fixed_indices] = self.P_in

        free_free_matrix = problem.M[free_indices][:, free_indices]
        free_fixed_matrix = problem.M[free_indices][:, fixed_indices]
        free_rhs = b[free_indices] - free_fixed_matrix @ p[fixed_indices]

        if free_indices.size > 0:
            free_pressures = spsolve(free_free_matrix, free_rhs)
            p[free_indices] = np.asarray(free_pressures, dtype=float)

        problem.p = p
        problem._update_node_pressures()
        problem._compute_edge_flows()

        return problem.p

    def _run_single_low_pressure_detailed(self, area_threshold=None):
        obstruction_state = self._sample_obstruction(area_threshold=area_threshold)
        problem = self._get_problem(obstruction_state)

        self._solve_with_input_pressure(problem)

        pressures = np.asarray(problem.p, dtype=float)
        monitor_indices = self._get_monitor_indices()

        if monitor_indices.size == 0:
            raise ValueError("no monitored nodes were selected")

        monitored_pressures = pressures[monitor_indices]
        min_pressure = float(np.min(monitored_pressures))
        failure_by_node = monitored_pressures < self.P_min
        failure = bool(np.any(failure_by_node))

        return pressures, min_pressure, failure, failure_by_node

    def estimate_pf_for_r(
        self,
        r_prob,
        n_iter=None,
        confidence=0.95,
        seed=None,
        area_threshold=None,
    ):
        self._ensure_ready()

        if n_iter is None:
            n_iter = self.n_samples

        n_iter = int(n_iter)

        if n_iter <= 0:
            raise ValueError("n_iter must be positive")

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.r_prob = float(r_prob)

        monitor_indices = self._get_monitor_indices()
        node_failure_counts = np.zeros(monitor_indices.size, dtype=int)
        min_pressures = []
        failures = 0

        for _ in range(n_iter):
            _, min_pressure, failure, failure_by_node = self._run_single_low_pressure_detailed(
                area_threshold=area_threshold
            )

            min_pressures.append(min_pressure)
            node_failure_counts += failure_by_node.astype(int)

            if failure:
                failures += 1

        p_fail = failures / n_iter
        z_value = norm.ppf(1 - (1 - confidence) / 2)
        margin = z_value * np.sqrt(p_fail * (1 - p_fail) / n_iter)

        monitor_nodes = [self.get_node_order()[idx] for idx in monitor_indices]
        node_failure_probability = {
            node: float(count / n_iter)
            for node, count in zip(monitor_nodes, node_failure_counts)
        }

        return {
            "r_prob": float(r_prob),
            "n_samples": n_iter,
            "P_fail": float(p_fail),
            "IC_lower": float(max(0.0, p_fail - margin)),
            "IC_upper": float(min(1.0, p_fail + margin)),
            "mean_min_pressure": float(np.mean(min_pressures)),
            "std_min_pressure": float(np.std(min_pressures)),
            "min_pressure_observed": float(np.min(min_pressures)),
            "node_failure_probability": node_failure_probability,
        }

    def solve_inverse_problem(
        self,
        n_iter=None,
        confidence=0.95,
        seed=42,
        use_upper_ci=False,
        area_threshold=None,
    ):
        self._ensure_ready()

        results = []

        for row in self.maintenance_table:
            stats = self.estimate_pf_for_r(
                r_prob=row["r_prob"],
                n_iter=n_iter,
                confidence=confidence,
                seed=seed,
                area_threshold=area_threshold,
            )

            criterion_value = stats["IC_upper"] if use_upper_ci else stats["P_fail"]
            accepted = criterion_value < self.x

            result = {
                "frequencia": row["frequencia"],
                "r_prob": row["r_prob"],
                "custo": row["custo"],
                "P_in": self.P_in,
                "P_min": self.P_min,
                "x": self.x,
                "criterion": "IC_upper" if use_upper_ci else "P_fail",
                "criterion_value": float(criterion_value),
                "accepted": bool(accepted),
                **stats,
            }

            results.append(result)

        accepted_results = [row for row in results if row["accepted"]]
        best_solution = accepted_results[0] if accepted_results else None

        self.p3_results = results
        self.best_solution = best_solution

        return best_solution, results

    def critical_r(self):
        if self.p3_results is None:
            raise ValueError("run solve_inverse_problem before calling critical_r")

        accepted = [row for row in self.p3_results if row["accepted"]]

        if not accepted:
            return None

        return max(row["r_prob"] for row in accepted)

    def summary(self):
        if self.p3_results is None:
            raise ValueError("run solve_inverse_problem before calling summary")

        return {
            "P_in": self.P_in,
            "P_min": self.P_min,
            "x": self.x,
            "best_solution": self.best_solution,
            "critical_r": self.critical_r(),
            "results": self.p3_results,
        }
