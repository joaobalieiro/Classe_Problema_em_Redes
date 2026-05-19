import numpy as np

from src.ProblemaP1 import ProblemaP1
from src.Grid import GridP1

def predial_params():
    mu = 1.0e-3                     # Pa*s (água ~ 20-25C)
    D = 0.010                       # 10 mm
    area = np.pi * (D**2) / 4.0     # m^2
    length = 1.0                    # m por aresta (trecho)
    Q_in = 2.0e-5                   # m^3/s (~1.2 L/min)
    patm = 0.0                      # Pa (manométrica)
    return dict(mu=mu, area=area, length=length, Q_in=Q_in, patm=patm)

def set_grid_p1(n, inflow_pressure=None):
    """
    Cria uma malha quadrada (n x n) do tipo GridP1 
    utilizando os parâmetros prediais padrão.
    
    Parâmetros:
    -----------
    n : int
        Número de nós em cada dimensão (linhas e colunas).
    inflow_pressure : float, opcional
        Pressão fixa de entrada para as condições de contorno.
        
    Retorno:
    --------
    GridP1
        Instância do problema configurada com as propriedades prediais.
    """
    params = predial_params()
    
    grid = GridP1(
        n=n,
        m=n,
        area=params['area'],
        length=params['length'],
        mu=params['mu'],
        patm=params['patm'],
        Q_in=params['Q_in'],
        inflow_pressure=inflow_pressure  # Propaga o parâmetro de pressão de entrada
    )
    
    return grid
