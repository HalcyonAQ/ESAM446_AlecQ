from scipy import sparse
import timesteppers as timesteppers
from timesteppers import StateVector, CrankNicolson, RK22
import finite as finite
import numpy as np

class Diffusionx:

    def __init__(self, c, D, d2x):
        self.X = timesteppers.StateVector([c], axis=0)
        N = c.shape[0]
        self.M = sparse.eye(N, N)
        self.L = -D*d2x.matrix


class Diffusiony:

    def __init__(self, c, D, d2y):
        self.X = timesteppers.StateVector([c], axis=1)
        N = c.shape[1]
        self.M = sparse.eye(N, N)
        self.L = -D*d2y.matrix
        
        
class DiffusionR:
    def __init__(self,c):
        N = len(c[0])
        self.X = timesteppers.StateVector([c])
        self.F = lambda X: X.data*(1 - X.data)


class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.c = c
        self.D = D
        self.dx2 = dx2
        self.dy2 = dy2
        self.iter = 0
        self.t = 0
            
    def step(self,dt):
        diffx = Diffusionx(self.c, self.D, self.dx2)
        diffy = Diffusiony(self.c, self.D, self.dy2)
        diffr = DiffusionR(self.c)
        ts_x = timesteppers.CrankNicolson(diffx, 0)
        ts_y = timesteppers.CrankNicolson(diffy, 1)
        rs = timesteppers.RK22(diffr)
        rs.step(1/2*dt)
        ts_y.step(1/2*dt)
        ts_x.step(dt)
        ts_y.step(1/2*dt)
        rs.step(1/2*dt)
        self.t = self.t + dt
        self.iter = self.iter+1

class Diffusionx_b:

    def __init__(self, u ,v, nu,spatial_order, domain):
        self.X = StateVector([u,v],axis = 0)
        self.u = self.X.variables[0]
        self.v = self.X.variables[1]
        self.x, self.y = domain.grids
        self.nu = nu
        self.spatial_order = spatial_order
        self.domain = domain
        self.d2x = finite.DifferenceUniformGrid(2,spatial_order,self.x,0)
        N = len(u)

        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = -nu*self.d2x.matrix
        L01 = Z
        L10 = Z
        L11 = -nu*self.d2x.matrix
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
class Diffusiony_b:

    def __init__(self, u ,v, nu,spatial_order, domain):
        self.X = StateVector([u,v], axis=1)
        self.u = self.X.variables[0]
        self.v = self.X.variables[1]
        self.x, self.y = domain.grids
        self.nu = nu
        self.spatial_order = spatial_order
        self.domain = domain
        self.d2y = finite.DifferenceUniformGrid(2,spatial_order,self.y,1)
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = -nu*self.d2y.matrix
        L01 = Z
        L10 = Z
        L11 = -nu*self.d2y.matrix
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
class DiffusionR_b:
    def __init__(self, u ,v, nu,spatial_order, domain):
        self.X = StateVector([u,v])
        self.u = self.X.variables[0]
        self.v = self.X.variables[1]
        x, y = domain.grids
        self.nu = nu
        self.spatial_order = spatial_order
        self.domain = domain
        dx = finite.DifferenceUniformGrid(1,spatial_order,x,0)
        dy = finite.DifferenceUniformGrid(1,spatial_order,y,1)
        self.F = lambda X: -np.append(X.variables[0]*(dx@X.variables[0])+X.variables[1]*(dy@X.variables[0]),X.variables[0]*(dx@X.variables[1])+X.variables[1]*(dy@X.variables[1]),axis=0)
        
class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.X = StateVector([u,v])
        self.u = self.X.variables[0]
        self.v = self.X.variables[1]
        self.x, self.y = domain.grids
        self.nu = nu
        self.spatial_order = spatial_order
        self.domain = domain
        self.iter = 0
        self.t = 0
       

    def step(self, dt):
        diffx = Diffusionx_b(self.u,self.v,self.nu,self.spatial_order,self.domain)
        diffy = Diffusiony_b(self.u,self.v,self.nu,self.spatial_order,self.domain)
        diffr = DiffusionR_b(self.u,self.v,self.nu,self.spatial_order,self.domain)
        ts_x = timesteppers.CrankNicolson(diffx, 0)
        ts_y = timesteppers.CrankNicolson(diffy, 1)
        rs = timesteppers.RK22(diffr)
        ts_y.step(1/2*dt)
        ts_x.step(1/2*dt)
        rs.step(dt)
        ts_x.step(1/2*dt)
        ts_y.step(1/2*dt)
        self.t = self.t + dt
        self.iter = self.iter+1


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

