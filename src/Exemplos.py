import numpy as np

from src.ProblemaP1 import ProblemaP1
from src.Grid import GridP1, GridP2

def predial_params():
    mu = 1.0e-3                     # Pa*s (água ~ 20-25C)
    D = 0.010                       # 10 mm
    area = np.pi * (D**2) / 4.0     # m^2
    length = 1.0                    # m por aresta (trecho)
    Q_in = 2.0e-5                   # m^3/s (~1.2 L/min)
    patm = 0.0                      # Pa (manométrica). Se quiser absoluta, use 101325.0
    return dict(mu=mu, area=area, length=length, Q_in=Q_in, patm=patm)

def set_grid_p1(n):
    """
    Cria uma malha quadrada (n x n) do tipo GridP1 
    utilizando os parâmetros prediais padrão.
    
    Parâmetros:
    -----------
    n : int
        Número de nós em cada dimensão (linhas e colunas).
        
    Retorno:
    --------
    GridP1
        Instância do problema configurada com as propriedades prediais.
    """
    # Carrega os parâmetros físicos e de fluxo
    params = predial_params()
    
    # Instancia o GridP1 passando n para as linhas e colunas (malha quadrada)
    grid = GridP1(
        n=n,
        m=n,
        area=params['area'],
        length=params['length'],
        mu=params['mu'],
        patm=params['patm'],
        Q_in=params['Q_in']
    )
    
    return grid

def set_grid_p2(n, r_prob, alpha, P_max, n_problems=100, n_samples=100, **kwargs):
    """
    Cria uma instância de GridP2 a partir de um GridP1 configurado 
    com parâmetros prediais, e adiciona os parâmetros estocásticos 
    para análise de Monte Carlo.
    
    Parâmetros:
    -----------
    n : int
        Número de nós em cada dimensão (linhas e colunas) para o GridP1.
    kwargs : dict
        Parâmetros adicionais para GridP2 (r_prob, alpha, P_max, n_problems, n_samples).
        
    Retorno:
    --------
    GridP2
        Instância do problema probabilístico configurada com as propriedades prediais e estocásticas.
    """
    # Cria o GridP1 base
    p1_instance = set_grid_p1(n)
    
    # Cria o GridP2 usando a instância do P1 e os parâmetros adicionais
    grid_p2 = GridP2(p1_instance=p1_instance, **kwargs)
    
    # Definindo parâmetros (reutilizando sua lógica)
    params = predial_params() 

    # 1. Cria a instância do GridP2 com parâmetros estocásticos
    grid_probabilistico = GridP2(
        n=5, m=5,
        area=params['area'],
        length=params['length'],
        mu=params['mu'],
        patm=params['patm'],
        Q_in=params['Q_in'],
        r_prob=r_prob,
        alpha=alpha,
        P_max=P_max,
    )

    # 2. Roda a análise de Monte Carlo (ProblemaP2)
    grid_probabilistico.run()

    # 3. Exibe o sumário estatístico
    resultados = grid_probabilistico.summary()
    print("Estimativa de Probabilidade de Falha:", resultados["P_f_hat"])
    print("Intervalo de Confiança (95%):", resultados["IC_95%"])

    # 4. Plota o estado inicial nominal (sem obstrução)
    grid_probabilistico.solve() # Resolve o cenário base para preencher os nós com as pressões
    grid_probabilistico.plot_grid()