import numpy as np
import copy
from scipy.stats import norm
from src.Except import *  # Mantido conforme original
from src.ProblemaP1 import ProblemaP1

class ProblemaP2(ProblemaP1):
    """
    Problema Probabilístico Direto.
    Herda ProblemaP1 e estima a probabilidade de falha P_f via Monte Carlo.
    """
    def __init__(
        self,
        nodes=None,
        edges=None,
        mu=None,
        patm=None,
        Q_ext=None,
        p1_instance=None,
        r_prob: float = 0.1,
        alpha: float = 2.0,
        P_max: float = 1e5,
        n_problems: int = 10,
        n_samples: int = 100,
        seed=None,
        inflow_pressure=None,
        **kwargs,
    ):
        if p1_instance is not None: 
            if not isinstance(p1_instance, ProblemaP1):
                raise TypeError("p1_instance deve ser uma instância de ProblemaP1.")
            
            self.__dict__.update(copy.deepcopy(p1_instance).__dict__)
            self._source_class = p1_instance.__class__
            
            if inflow_pressure is not None:
                self.inflow_pressure = inflow_pressure
        else:
            super().__init__(
                nodes=nodes, 
                edges=edges, 
                mu=mu, 
                patm=patm, 
                Q_ext=Q_ext, 
                inflow_pressure=inflow_pressure, 
                **kwargs
            )

        self.r_prob     = r_prob
        self.alpha      = alpha
        self.P_max      = P_max
        self.n_problems = n_problems
        self.n_samples  = n_samples
        self.rng        = np.random.default_rng(seed)
        self.thetas: np.ndarray | None = None  

    def _sample_obstruction(self):
        # Correção: Amostra diretamente do get_edge_list() para evitar desalinhamento de chaves
        return {
            edge: self.rng.random() < self.r_prob
            for edge in self.get_edge_list()
        }

    def _get_scenario(self, obstruction_state: dict) -> "ProblemaP1":
        _exclude = {"r_prob", "alpha", "P_max", "n_problems", "n_samples", "rng", "thetas"}
        
        cls = getattr(self, '_source_class', ProblemaP1)
        scenario = cls.__new__(cls)
        
        scenario.__dict__.update(
            copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in _exclude})
        )

        edge_index = {edge: idx for idx, edge in enumerate(self.get_edge_list())}
        scenario.setup()
        K_lil = scenario.K.tolil()
        
        for edge, obstructed in obstruction_state.items():
            if obstructed:
                idx = edge_index[edge]
                K_lil[idx, idx] /= self.alpha
        
        scenario.K = K_lil.tocsr()
        scenario.M = (scenario.A.T @ scenario.K @ scenario.A).tocsr()
        scenario.is_fitted = True
        return scenario

    def get_scenario_example(self, grid=False) -> "ProblemaP1":
        obstruction_state = self._sample_obstruction()
        return self._get_scenario(obstruction_state) 

    def _run_single_simulation(self) -> int:
        scenario = self._get_scenario(self._sample_obstruction())
        # Correção: Removido o scenario.setup() daqui, que resetava as obstruções aplicadas
        scenario.solve()
        return int(np.any(scenario.p > self.P_max))

    def _run_problem(self) -> float:
        indicators = np.array([self._run_single_simulation() for _ in range(self.n_samples)])
        return indicators.mean()

    def run(self):
        self.thetas = np.array([self._run_problem() for _ in range(self.n_problems)])

    def probability_of_failure(self) -> float:
        self._check_fitted()
        return float(self.thetas.mean())

    def std_estimator(self) -> float:
        self._check_fitted()
        if self.n_problems < 2:
            return 0.0
        return float(self.thetas.std(ddof=1))

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        self._check_fitted()
        pf  = self.probability_of_failure()
        if self.n_problems < 2:
            return (pf, pf)
        se  = self.std_estimator() / np.sqrt(self.n_problems)
        z   = norm.ppf(1 - (1 - confidence) / 2)
        return (max(0.0, pf - z * se), min(1.0, pf + z * se))

    def summary(self, confidence: float = 0.95) -> dict:
        self._check_fitted()
        pf  = self.probability_of_failure()
        std = self.std_estimator()
        se  = std / np.sqrt(self.n_problems) if self.n_problems >= 2 else 0.0
        ci  = self.confidence_interval(confidence)

        return {
            "n_problems":   self.n_problems,
            "n_samples":    self.n_samples,
            "P_f_hat":      pf,
            "std_between":  std,
            "std_error":    se,
            f"IC_{int(confidence*100)}%": ci,
            "thetas":       self.thetas.tolist(),
        }

    def _check_fitted(self):
        if self.thetas is None:
            raise RuntimeError("Execute run() antes de acessar os resultados.")
