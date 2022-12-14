from scipy import sparse
import timesteppers as timesteppers
from timesteppers import StateVector, CrankNicolson, RK22
import finite as finite
import numpy as np

class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J


class BurgersFI:
    
    def __init__(self, u, nu, spatial_order, grid):
        self.X = timesteppers.StateVector([u])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        d = finite.DifferenceUniformGrid(1, spatial_order, grid)
        self.N = len(u)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d.matrix @ X.data)
        self.F = f
        
        def J(X):
            matr_u = sparse.diags(X.data)
            return  -d.matrix@matr_u - matr_u@d.matrix
        self.J = J


class ReactionTwoSpeciesDiffusion: 
    def __init__(self, X, D, r, spatial_order, grid):
        self.X = X
        N = len(X.variables[0])
        I = sparse.eye(N,N)
        Z = sparse.csr_matrix((N, N))
        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        L00 = -D*d2.matrix
        L01 = Z
        L10 = Z
        L11 = -D*d2.matrix
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        def F(X):
            row_1 = X.data[0:N]*(1 - X.data[0:N] - X.data[N:2*N])
            row_2 = r*X.data[N:2*N]*(X.data[0:N]-X.data[N:2*N])

            return np.concatenate((row_1,row_2),axis=0)
        self.F = F
        def J(X):
            c1 = sparse.diags(X.data[0:N])
            c2 = sparse.diags(X.data[N:2*N])
            J00 = I - 2*c1 - c2
            J01 = -c1
            J10 = r*c2
            J11 = r*c1 - 2*r*c2
            return sparse.bmat([[J00, J01],
                              [J10, J11]]) 
        self.J = J





class DiffusionBC_x:
    def __init__(self, c, D, spatial_order, domain):
        self.X = timesteppers.StateVector([c], axis=0)
        x = domain.grids[0]
        N = c.shape[0]
        d2x = finite.DifferenceUniformGrid(2,spatial_order, x,0)
        self.M = sparse.eye(N, N)
        self.M = self.M.tocsr()
        self.M[0,:] = 0
        self.M[-1,:] = 0
        
        self.L = -D*d2x.matrix
        self.L = self.L.tocsr()
        self.L[0,:] = 0
        self.L[0,0] = 1
        self.L[-1,:] = 0
        self.L[-1,-1] = 3/(2*x.dx)
        self.L[-1,-2] = -2/x.dx
        self.L[-1,-3] = 1/(2*x.dx)


class DiffusionBC_y:

    def __init__(self, c, D, spatial_order, domain):
        self.X = timesteppers.StateVector([c], axis=1)
        N = c.shape[1]
        y = domain.grids[1]
        d2y = finite.DifferenceUniformGrid(2,spatial_order, y,1)
        self.M = sparse.eye(N, N)
        self.L = -D*d2y.matrix

class DiffusionBC:
    def __init__(self, c,D, spatial_order, domain):
        self.X = StateVector([c])
        self.c = c
        self.D = D
        self.x,self.y = domain.grids
        self.spatial_order = spatial_order
        self.domain = domain
        self.iter = 0
        self.t = 0


    def step(self, dt):
        diffx = DiffusionBC_x(self.c,self.D,self.spatial_order,self.domain)
        diffy = DiffusionBC_y(self.c,self.D,self.spatial_order,self.domain)

        ts_x = timesteppers.CrankNicolson(diffx,0)
        ts_y = timesteppers.CrankNicolson(diffy,1)
        ts_x.step(dt/2)
        ts_y.step(dt)
        ts_x.step(dt/2)

        self.t += dt
        self.iter += 1


class DWu:
    def __init__(self,u, v, p, spatial_order, domain):
        self.u = u
        self.v = v
        self.p = p
        N = len(u)
        self.X= StateVector([u])
        dx= finite.DifferenceUniformGrid(1,spatial_order,domain.grids[0],0)
        self.F = lambda X : dx@-p
        def BC(X):
            u = X.data
            u[-1,:] = 0
            u[0,:] = 0
        self.BC = BC
        
        
class DWv:
    def __init__(self,u, v, p, spatial_order, domain):
        self.u = u
        self.v = v
        self.p = p
        N = len(u)
        self.X= StateVector([v])
        dy= finite.DifferenceUniformGrid(1,spatial_order,domain.grids[1],1)
        self.F = lambda X : dy@-p
        
        
class DWp:
    def __init__(self,u, v, p, spatial_order, domain):
        un = u
        un[-1,:] = 0
        un[0,:] = 0
        self.v = v
        self.p = p
        N = len(u)
        self.X= StateVector([p])
        dx= finite.DifferenceUniformGrid(1,spatial_order,domain.grids[0],0)
        dy= finite.DifferenceUniformGrid(1,spatial_order,domain.grids[1],1)
        self.F = lambda X : dx@-un + dy@-v
        
class Wave2DBC:
    def __init__(self,u, v, p, spatial_order, domain):
        self.X = timesteppers.StateVector([u,v,p])
        self.u = u
        self.v = v
        self.p = p
        self.spatial_order = spatial_order
        self.domain = domain
        self.x, self.y = domain.grids
        self.iter = 0
        self.t = 0
    def step(self, dt):
        diffu = DWu(self.u,self.v,self.p,self.spatial_order,self.domain)
        diffv = DWv(self.u,self.v,self.p,self.spatial_order,self.domain)
        diffp = DWp(self.u,self.v,self.p,self.spatial_order,self.domain)
        
        su = timesteppers.RK22(diffu)
        sv = timesteppers.RK22(diffv)
        sp = timesteppers.RK22(diffp)
        
        su.step(1/2*dt)
        sv.step(1/2*dt)
        sp.step(dt)
        sv.step(1/2*dt)
        su.step(1/2*dt)
        
        self.t = self.t +dt
        self.iter = self.iter +1
        

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
        self.u = u
        self.v = v
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
        self.u = u
        self.v = v
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
        self.X = StateVector([u,v],0)
        N = len(u)
        self.u = u
        self.v = v
        self.x, self.y = domain.grids
        self.nu = nu
        self.spatial_order = spatial_order
        self.domain = domain
        dx = finite.DifferenceUniformGrid(1,spatial_order,domain.grids[0],0)
        dy = finite.DifferenceUniformGrid(1,spatial_order,domain.grids[1],1)
        
        f = lambda X: -np.concatenate(((X.data[:N,:] * (dx@X.data[:N,:])+ X.data[N:,:] * (dy @ X.data[:N,:]),X.data[:N,:]*(dx @ X.data[N:,:])+X.data[N:,:]*(dy@X.data[N:,:]))),axis = 0)
        self.F = f
        
        
class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.X = StateVector([u,v])
        self.u = u
        self.v = v
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
        rs.step(1/2*dt)
        ts_y.step(1/2*dt)
        ts_x.step(dt)
        ts_y.step(1/2*dt)
        rs.step(1/2*dt)
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
