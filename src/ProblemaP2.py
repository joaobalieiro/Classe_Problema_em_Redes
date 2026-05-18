import numpy as np
import copy
from scipy.stats import norm
from src.Except import *
from src.ProblemaP1 import ProblemaP1


class ProblemaP2(ProblemaP1):
    """
    Problema Probabilístico Direto.
    Herda ProblemaP1 e estima a probabilidade de falha P_f via Monte Carlo.

    Estrutura:
        - n_problems : número de instâncias P1 independentes (experimentos)
        - n_samples  : número de simulações MC por instância

    Para cada instância k in {1,...,n_problems}, geram-se n_samples cenários
    aleatórios de obstrução e calcula-se o estimador local:

        theta_k = (1 / n_samples) * sum_{j=1}^{n_samples} I_j^{(k)}

    A estimativa final de P_f é a média dos estimadores locais:

        P_f_hat = (1 / n_problems) * sum_{k=1}^{n_problems} theta_k
    """

    def __init__(
        self,
        *args,
        p1_instance=None,
        r_prob: float = 0.1,
        alpha: float = 2.0,
        P_max: float = 1e5,
        n_problems: int = 10,
        n_samples: int = 100,
        seed=None,
        **kwargs,
    ):
        """
        Parâmetros
        ----------
        p1_instance : ProblemaP1, opcional
            Instância base de P1. Se fornecida, a rede é copiada dela.
        r_prob      : float
            Probabilidade de obstrução de cada aresta.
        alpha       : float
            Fator de penalização da condutância (alpha > 1).
        P_max       : float
            Pressão máxima admissível.
        n_problems  : int
            Número de instâncias P1 independentes (experimentos MC).
        n_samples   : int
            Número de simulações por instância.
        seed        : int ou None
            Semente para reprodutibilidade.
        """
       
        if p1_instance is not None: 
            if not isinstance(p1_instance, ProblemaP1): # Valida se o objeto fornecido é realmente do tipo 'ProblemaP1'
                raise TypeError("p1_instance deve ser uma instância de ProblemaP1.")
            
            # 1. copy.deepcopy(p1_instance): Cria uma cópia totalmente independente do objeto original.
            # 2. .__dict__: Pega o dicionário com todos os atributos dessa cópia.
            # 3. self.__dict__.update(...): Injeta esses atributos na instância atual (self).
            # O resultado final é que o 'self' se torna uma cópia exata e segura de 'p1_instance'.
            self.__dict__.update(copy.deepcopy(p1_instance).__dict__)
        else:
            super().__init__(*args, **kwargs)

        self.r_prob     = r_prob
        self.alpha      = alpha
        self.P_max      = P_max
        self.n_problems = n_problems
        self.n_samples  = n_samples
        self.rng        = np.random.default_rng(seed)

        # Resultados preenchidos após run()
        self.thetas: np.ndarray | None = None   # estimadores locais theta_k

    def _sample_obstruction(self):
        """
        Sorteia independentemente o estado de cada aresta.
        Retorna dict {(u,v): bool}.
        """
        return {
            (u, v): self.rng.random() < self.r_prob
            for u in self.edges
            for v in self.edges[u]
        }

    def _get_scenario(self, obstruction_state: dict) -> "ProblemaP1":
        """
        Constrói uma instância P1 com condutâncias modificadas pelo
        estado de obstrução sorteado.
        """
        _exclude = {
            "r_prob", "alpha", "P_max", "n_problems",
            "n_samples", "rng", "thetas",
        }
        scenario = ProblemaP1.__new__(ProblemaP1)
        scenario.__dict__.update(
            copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in _exclude})
        )

        edge_index = {edge: idx for idx, edge in enumerate(self.get_edge_list())}

        scenario.K = scenario.K.tolil(copy=True)
        for edge, obstructed in obstruction_state.items():
            if obstructed:
                idx = edge_index[edge]
                scenario.K[idx, idx] /= self.alpha

        scenario.K = scenario.K.tocsr()
        scenario.S = (scenario.A.T @ scenario.K @ scenario.A).tocsr()
        scenario.is_fitted = True
        return scenario

    def _run_single_simulation(self) -> int:
        """
        Executa uma simulação MC.
        Retorna 1 (falha) ou 0 (sem falha).
        """
        scenario = self._get_scenario(self._sample_obstruction())
        scenario.solve()
        return int(np.any(scenario.p > self.P_max))

    def _run_problem(self) -> float:
        """
        Executa n_samples simulações para uma instância P1.
        Retorna o estimador local theta_k = média dos indicadores.
        """
        indicators = np.array([self._run_single_simulation() for _ in range(self.n_samples)])
        return indicators.mean()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def run(self):
        """
        Executa n_problems instâncias, cada uma com n_samples simulações.
        Armazena os estimadores locais em self.thetas.
        """
        self.thetas = np.array([self._run_problem() for _ in range(self.n_problems)])

    def probability_of_failure(self) -> float:
        """
        Estimativa de P_f como média dos estimadores locais:
            P_f_hat = (1/n_problems) * sum_k theta_k
        """
        self._check_fitted()
        return float(self.thetas.mean())

    def std_estimator(self) -> float:
        """
        Desvio padrão amostral dos estimadores locais theta_k.
        Mede a variabilidade entre os n_problems experimentos.
        """
        self._check_fitted()
        return float(self.thetas.std(ddof=1))

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        Intervalo de confiança para P_f baseado na distribuição dos theta_k.

        Usa a aproximação normal:
            IC = P_f_hat ± z * std(theta_k) / sqrt(n_problems)
        """
        self._check_fitted()
        pf  = self.probability_of_failure()
        se  = self.std_estimator() / np.sqrt(self.n_problems)
        z   = norm.ppf(1 - (1 - confidence) / 2)
        return (max(0.0, pf - z * se), min(1.0, pf + z * se))

    def summary(self, confidence: float = 0.95) -> dict:
        """
        Resumo estatístico dos resultados.

        Retorna
        -------
        dict com:
            n_problems          : número de instâncias P1
            n_samples           : simulações por instância
            P_f_hat             : estimativa de P_f (média dos theta_k)
            std_between         : desvio padrão entre os theta_k
            std_error           : erro padrão da média
            IC                  : intervalo de confiança (lower, upper)
            thetas              : vetor com todos os estimadores locais
        """
        self._check_fitted()
        pf  = self.probability_of_failure()
        std = self.std_estimator()
        se  = std / np.sqrt(self.n_problems)
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

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.thetas is None:
            raise RuntimeError("Execute run() antes de acessar os resultados.")