import numpy as np
import math
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

class Timestepper:

    def __init__(self, u, f):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper):

    def _step(self, dt):
        return self.u + dt*self.func(self.u)


class LaxFriedrichs(Timestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.func(self.u)


class Leapfrog(Timestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.func(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.func(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, func1, func2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(Timestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        k = np.array([np.zeros(len(self.u)) for x in range(self.stages)])
        i = 0
        while i < self.stages:
            k[i] = self.func(self.u+dt*np.sum(np.transpose([x*y for x,y in zip(k,self.a[i])]),axis = 1))
            i = i+1
            
        return self.u+dt*np.sum(np.transpose([x*y for x,y in zip(k,self.b)]),axis = 1)



class AdamsBashforth(Timestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.uf = []
        self.uf.append([self.u])

    def _step(self, dt):
        if self.iter < self.steps:
            u_f=[]
            u_f.append(self.u)
            s = np.array([np.zeros(self.iter) for x in range(self.iter)])
            b = np.zeros(self.iter)
            i = 0
            while i<self.iter:
                j = 0
                while j<self.iter:
                    s[j][i] = (-i)**j/math.factorial(j)
                    j=j+1
                b[i] = 1/math.factorial(i+1)
                i = i+1
            a = np.zeros(self.iter)
            if self.iter != 0:
                a = np.linalg.inv(s).dot(b)
            
            i = 0
            while i<self.iter:
                u_f.append(self.dt*a[i]*self.func(self.uf[self.iter-i]))
                i = i+1
            self.uf.append(np.sum(np.transpose(u_f),axis=1))
            return np.sum(np.transpose(u_f),axis=1)
        else:
            u_f = []
            u_f.append(self.uf[self.iter])
            s = np.array([np.zeros(self.steps) for x in range(self.steps)])
            b = np.zeros(self.steps)
            i = 0
            while i<self.steps:
                j = 0
                while j<self.steps:
                    s[j][i] = (-i)**j/math.factorial(j)
                    j=j+1
                b[i] = 1/math.factorial(i+1)
                i = i+1
            a = np.zeros(self.steps)
            if self.iter != 0:
                a = np.linalg.inv(s).dot(b)
            
            i = 0
            while i<self.steps:
                u_f.append(self.dt*a[i]*self.func(self.uf[self.iter-i]))
                i = i+1
            self.uf.append(np.sum(np.transpose(u_f),axis=1))
            return np.sum(np.transpose(u_f),axis=1)
        
class BackwardEuler(Timestepper):

    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.u)


class CrankNicolson(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.func.matrix
            self.RHS = self.I + dt/2*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.RHS @ self.u)


class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)
        self.steps = steps
        self.uf = []
        self.uf.append(self.u)
        self.dt = []

    def _step(self, dt):
        if self.iter == 0:
            self.dt.append(dt)
        if self.iter+1 < self.steps:
            self.dt.append(dt)
            k = self.iter+2
            s = np.array([np.zeros(k) for x in range(k)])
            i = 0
            while i<k:
                j = 0
                while j<k:
                    s[j][i] = (-i*self.dt[-1-i])**j/math.factorial(j)
                    j=j+1
                i = i+1
            print(s)
            a = np.zeros(k)
            b = np.zeros(k)
            b[1] = dt
            a = np.linalg.inv(s).dot(b)
            a0 = 1/a[0]
            a = a*a0
            LHS = (self.I-a0*dt*self.func.matrix).A
            RHS = []
            l = 1
            while l<k:
                RHS.append(-a[l]*self.uf[-l])
                l = l+1 
            R = np.sum(np.transpose(RHS),axis=1)
            unew = np.linalg.solve(LHS,R)
            self.uf.append(unew)
            return unew
        else:
            self.dt.append(dt)
            k = self.steps+1
            s = np.array([np.zeros(k) for x in range(k)])
            i = 0
            while i<k:
                j = 0
                while j<k:
                    s[j][i] = (-i*self.dt[-1-i])**j/math.factorial(j)
                    j=j+1
                i = i+1
            print(s)
            a = np.zeros(k)
            b = np.zeros(k)
            b[1] = dt
            a = np.linalg.inv(s).dot(b)
            a0 = 1/a[0]
            a = a*a0
            LHS = (self.I-a0*dt*self.func.matrix).A
            RHS = []
            l = 1
            while l<self.steps+1:
                RHS.append(-a[l]*self.uf[-l])
                l = l+1
            R = np.sum(np.transpose(RHS),axis=1)
            unew = np.linalg.solve(LHS,R)
            self.uf = self.uf[1:]
            self.dt = self.dt[1:]
            self.uf.append(unew)
            return unew

