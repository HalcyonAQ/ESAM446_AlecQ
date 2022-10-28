import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque
import math

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

        self.u_list = []
        self.K_list = []
        for i in range(self.stages):
            self.u_list.append(np.copy(u))
            self.K_list.append(np.copy(u))

    def _step(self, dt):
        u = self.u
        u_list = self.u_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(u_list[0], u)
        for i in range(1, stages):
            K_list[i-1] = self.func(u_list[i-1])

            np.copyto(u_list[i], u)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                u_list[i] += self.a[i, j]*dt*K_list[j]

        K_list[-1] = self.func(u_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            u += self.b[i]*dt*K_list[i]

        return u


class AdamsBashforth(Timestepper):

    def __init__(self, u, L_op, steps, dt):
        super().__init__(u, L_op)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(u))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.func(self.u)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.u += self.dt*coeff*self.f_list[i].data
        return self.u

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


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
        self.N = N
        self.I = sparse.eye(N, N)
        self.steps = steps
        self.uf = []
        self.dt = []

    def _step(self, dt):
        if self.iter+1 < self.steps:
            self.dt.append(dt)
            k = self.iter+2
            S = np.zeros((k,k))
            S[0,0] = 1

            dt_real = np.zeros(len(self.dt)+1)
            for i in range(len(self.dt)):
                dt_real[i] = np.sum(self.dt[i:])
            dt_r = list(reversed(dt_real))
            i = 1
            while i<k:
                j = 0
                while j<k:
                    S[j,i] = ((-1*dt_r[i])**j)/math.factorial(j)
                    j = j+1
                i = i+1
            b = np.zeros(k)
            b[1] = 1
            a = np.linalg.inv(S).dot(b)
            
            
            self.uf.append(self.u)
            rev = list(reversed(self.uf))
            LHS = self.func.matrix - a[0]* self.I
            RHS = np.zeros(self.N)
            l = 1
            while l<len(a):
                RHS+=a[l]*rev[l-1]
                l = l+1
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            return self.LU.solve(RHS)

        else:
            self.dt.append(dt)
            k = self.steps+1
            S = np.zeros((k,k))
            S[0,0] = 1

            dt_real = np.zeros(len(self.dt)+1)
            for i in range(len(self.dt)):
                dt_real[i] = np.sum(self.dt[i:])
            dt_r = list(reversed(dt_real))
            
            i = 1
            while i<k:
                j = 0
                while j<k:
                    S[j,i] = ((-1*dt_r[i])**j)/math.factorial(j)
                    j = j+1
                i = i+1
            b = np.zeros(k)
            b[1] = 1
            a = np.linalg.inv(S).dot(b)
            
            
            self.uf.append(self.u)
            rev = list(reversed(self.uf))
            LHS = self.func.matrix - a[0]* self.I
            RHS = np.zeros(self.N)
            l = 1
            while l<len(a):
                RHS+=a[l]*rev[l-1]
                l = l+1
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = self.dt[1:]
            self.uf = self.uf[1:]
            return self.LU.solve(RHS)
        
class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])


class IMEXTimestepper:

    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):
    

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.xf = []
        self.xs = []
        self.xs.append(self.X)
        self.xf.append(self.X.data)
    def _step(self, dt):
        if self.iter+1 < self.steps:

            ##calculate ai
            k = self.iter+2
            s = np.array([np.zeros(k) for x in range(k)])
            i = 0
            while i<k:
                j = 0
                while j<k:
                    s[j][i] = (-i*dt)**j/math.factorial(j)
                    j=j+1
                i = i+1

            a = np.zeros(k)
            d = np.zeros(k)
            d[1] = 1
            a = np.linalg.inv(s).dot(d)
            a0 = a[0]
            ##calculate bi
            k = self.iter+1
            s = np.array([np.zeros(k) for x in range(k)])
            i = 1
            while i<k+1:
                j = 0
                while j<k:
                    s[j][i-1] = (-i*dt)**j/math.factorial(j)
                    j=j+1
                i = i+1
         
            b = np.zeros(k)
            d = np.zeros(k)
            d[0] = 1
            b = np.linalg.inv(s).dot(d)
            ##rewrite the equation
            LHS = (self.M*a0+self.L)

            RHS1 = []
            RHS2 = []
            l = 1
            k = self.iter+2
            while l<k:
                RHS1.append(-a[l]*self.xf[-l]*self.M)
                RHS2.append(b[l-1]*self.F(self.xs[-l]))
                l = l+1 
            R = np.sum(np.transpose(RHS1+RHS2),axis=1)
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.xf.append(LU.solve(R))
            self.xs.append(StateVector(np.array([LU.solve(R)])))
            return LU.solve(R)
        
        else:

            ##calculate ai
            k = self.steps+1
            s = np.array([np.zeros(k) for x in range(k)])
            i = 0
            while i<k:
                j = 0
                while j<k:
                    s[j][i] = (-i*dt)**j/math.factorial(j)
                    j=j+1
                i = i+1
            a = np.zeros(k)
            d = np.zeros(k)
            d[1] = 1
            a = np.linalg.inv(s).dot(d)
            a0 = a[0]

            
            ##calculate bi
            k = self.steps
            s = np.array([np.zeros(k) for x in range(k)])
            i = 1
            while i<k+1:
                j = 0
                while j<k:
                    s[j][i-1] = (-i*dt)**j/math.factorial(j)
                    j=j+1
                i = i+1
            b = np.zeros(k)
            d = np.zeros(k)
            d[0] = 1
            b = np.linalg.inv(s).dot(d)

            ##rewrite the equation
            LHS = (self.M*a0+self.L)
            RHS1 = []
            RHS2 = []
            l = 1
            k = self.steps+1
            while l<k:
                RHS1.append(-a[l]*self.xf[-l]*self.M)
                RHS2.append(b[l-1]*self.F(self.xs[-l]))
                l = l+1 
            R = np.sum(np.transpose(RHS1+RHS2),axis=1)
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.xf = self.xf[1:]
            self.xf.append(LU.solve(R))
            self.xs = self.xs[1:]
            self.xs.append(StateVector(np.array([LU.solve(R)])))
            return LU.solve(R)
