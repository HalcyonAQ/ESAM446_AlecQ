from timesteppers import StateVector
from scipy import sparse
import numpy as np

class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))
        self.d = d
        self.rho0 = rho0
        self.p0 = gammap0
 
        if type(rho0) != int and type(rho0)!=float:
            self.rho0 = sparse.diags(rho0).A
        if type(gammap0) != int and type(gammap0)!=float:
            self.p0 = sparse.diags(gammap0).A
        
        
        M00 = self.rho0*I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = self.d.matrix
        L10 = self.p0*self.d.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N = len(c)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))
        self.D = D
        self.d2 = d2.matrix
        
       
        self.M = I
        self.L = -self.D*self.d2
        self.F = lambda X: X.data*(c_target - X.data)

