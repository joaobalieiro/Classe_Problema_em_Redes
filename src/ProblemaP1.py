import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from src.Except import *
from src.Grafo import *

class ProblemaP1(Grafo):
    def __init__(self, nodes, edges, mu, patm, Q_ext=None):
        """
        nodes: lista ou dict de nós
        edges: dict de edges {u:{v:{'area':..., 'length':...}}}
        mu: viscosidade
        patm: pressão de referência
        Q_ext: dict {node: fluxo externo}
        """
        super().__init__(nodes, edges, kind='Directed')
        self.mu = mu
        self.Q_ext = None if Q_ext is None else Q_ext
        self.patm = patm
        self.validate_problem_schema()

        self.A = None
        self.M = None
        self.K = None
        self.p = None
        self.q = None
        self.is_fitted = False

        if Q_ext is not None:
            self.set_Q_ext(Q_ext)  

    def solve(self):
        if self.Q_ext is None:
            raise ValueError("Define Q_ext before solving the system of equations. Use set_Q_ext() method.")

        self.K = self._compute_physical_matrix()
        self.A = self.compute_connection_matrix(sparse_output=True)
        self.M = (self.A.T @ self.K @ self.A).tocsr()
        self.is_fitted = True

        node_keys = self.get_node_order()
        b = np.array([self.Q_ext[key] for key in node_keys], dtype=float)

        fixed_indices = np.array([i for i, ext_flow in enumerate(b) if ext_flow < 0], dtype=int)
        if fixed_indices.size == 0:
            raise ValueError("No node with negative flow found for reference pressure.")

        n = len(node_keys)
        all_indices = np.arange(n, dtype=int)
        free_indices = np.setdiff1d(all_indices, fixed_indices)

        p = np.zeros(n, dtype=float)
        p[fixed_indices] = self.patm

        free_free_matrix = self.M[free_indices][:, free_indices]
        free_fixed_matrix = self.M[free_indices][:, fixed_indices]
        free_rhs = b[free_indices] - free_fixed_matrix @ p[fixed_indices]

        if free_indices.size > 0:
            free_pressures = spsolve(free_free_matrix, free_rhs)
            p[free_indices] = np.asarray(free_pressures, dtype=float)

        self.p = p
        self._update_node_pressures()
        self._compute_edge_flows()
        return self.p
    
    def _update_node_pressures(self):
        for idx, key in enumerate(self.get_node_order()):
            self.nodes[key]["pressao"] = float(self.p[idx])

    def _compute_edge_flows(self):
        if self.p is None:
            raise ValueError("solve must be called before computing edge flows.")
        if self.A is None or self.K is None:
            raise NotFittedError()

        pressure_drop = self.A @ self.p
        self.q = np.asarray(self.K @ pressure_drop, dtype=float).reshape(-1)

        for idx, (u, v) in enumerate(self.get_edge_list()):
            self.edges[u][v]["vazao"] = float(self.q[idx])
            self.edges[u][v]["delta_pressao"] = float(pressure_drop[idx])

        return self.q    

    def set_Q_ext(self, Q_ext: dict):
        """
        Define Q_ext como um dicionário {node: fluxo} e valida:
        - todos os nós estão presentes
        - soma dos fluxos é ~0
        """
        if not isinstance(Q_ext, dict):
            raise TypeError("Q_ext must be a dictionary {node: flow}.")
        
        missing_nodes = [node for node in self.nodes if node not in Q_ext]
        if missing_nodes:
            raise ValueError(f"Q_ext missing values for nodes: {missing_nodes}")
        
        total_flux = sum(Q_ext.values())
        if abs(total_flux) > 1e-6:
            raise ValueError(f"External flux does not conserve mass. Sum(Q_ext)={total_flux}")

        self.Q_ext = Q_ext
    
        for node, q in Q_ext.items():
                self.nodes[node]["fluxo_externo"] = float(q)
        
    def validate_problem_schema(self):
        if not isinstance(self.edges, dict):
            raise TypeError("Edges must be a dictionary.")

        for u in self.edges:
            if not isinstance(self.edges[u], dict):
                raise TypeError(f"Edges from node {u} must be a dictionary.")

            for v, attrs in self.edges[u].items():

                # checar se attrs é dict
                if not isinstance(attrs, dict):
                    raise TypeError(f"Edge ({u}->{v}) must have attribute dictionary.")

                # checar chaves obrigatórias
                required_keys = ["area", "length"]
                for key in required_keys:
                    if key not in attrs:
                        raise ValueError(f"Edge ({u}->{v}) missing required attribute '{key}'.")

                A = attrs["area"]
                L = attrs["length"]

                # checar tipo numérico
                if not isinstance(A, (int, float)):
                    raise TypeError(f"Edge ({u}->{v}) area must be numeric.")
                if not isinstance(L, (int, float)):
                    raise TypeError(f"Edge ({u}->{v}) length must be numeric.")

                # checar valores físicos válidos
                if A <= 0:
                    raise ValueError(f"Edge ({u}->{v}) area must be > 0.")
                if L <= 0:
                    raise ValueError(f"Edge ({u}->{v}) length must be > 0.")

        return True
    
    def assert_solvability(self, tol=1e-6):
        total_flux = sum(self.Q_ext.values())
        if abs(total_flux) > 1e-6:
            raise ValueError(f"External flux does not conserve mass. Sum(Q_ext)={total_flux}")

    def _compute_conductivity(self, area, length):
        if self.mu <= 0:
            raise ValueError("mu<=0")
        D = np.sqrt(4 * area / np.pi)
        return (np.pi * D**4) / (128 * self.mu * length)

    def _compute_physical_matrix(self):
        diagonal = []

        for u in self.edges:
            for v, attrs in self.edges[u].items():
                A = float(attrs["area"])
                L = float(attrs["length"])
                diagonal.append(self._compute_conductivity(A, L))

        self.K = diags(diagonal, offsets=0, format="csr", dtype=float)
        return self.K
    
    def validate_Q_ext(self):
        if len(self.Q_ext) != self.num_nodes:
            raise IndexError("Q_ext should be the same size as the number of nodes.")

    # override   
    def insert_edge(self, u, v, **kwargs):
        if u not in self.nodes or v not in self.nodes:
            raise Exception(f"Nodes {u} and {v} must exist.")
        
        required_fields = ["area", "length"]
        missing_fields = [field for field in required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required edge attributes: {', '.join(missing_fields)}")

        if self.kind == 'Directed':
            if v in self.edges and u in self.edges[v]:
                raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")

        if u not in self.edges:
            self.edges[u] = {}

        if v not in self.edges[u]:
            self.num_edges += 1

        self.edges[u][v] = kwargs
        
    
