import numpy as np
from scipy.special import factorial
from scipy import sparse
import math

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class DifferenceUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.grid = grid
        
        
        ##define needed variables
        size = int(2*math.floor((self.derivative_order+1)/2)+self.convergence_order-1)
        
        j = np.arange(int(-(size+1)/2+1),int((size+1)/2))
        b = np.zeros(size,dtype = int)
        b[self.derivative_order] = 1
        S = []
        
        ##create s matrix
        i = 0
        while i<size:
            S.append(list(1/factorial(i)*(j*self.grid.dx)**i))
            i = i+1
            
        ##compute a matrix from s matrix
        a = np.linalg.inv(S) @ b
        
        ##assign values to sparse matrix
        D = sparse.diags(a, offsets=j, shape = [int(self.grid.N),int(self.grid.N)])
        D = D.tocsr()
        
        ##impute corner cases
        p = 0
        q = 0
        r = 0
        while p < size:
            while q < size:
                D[p,j[q]] = a[r]
                D[-(p+1),-j[q]-1] = a[-r-1]
                q = q+1
                r = r + 1
            p = p + 1
            q = p
            r = 0
        
        
        self.matrix = D

    def __matmul__(self, other):
        return self.matrix @ other
    


class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.grid = grid
        
        
        c = math.floor(self.convergence_order/2)
        
        j = np.arange(-c,c+1)
        b = np.zeros(2*c+1,dtype = int)
        b[self.derivative_order] = 1
        S = []
        i = 0
        while i<2*c+1:
            S.append(list(1/factorial(i)*(j*self.grid.dx)**i))
            i = i+1
        a = np.linalg.inv(S) @ b
        D = sparse.diags(a, offsets=j, shape = [5,5])
        D = D.tocsr()
        p = 0
        while p < len(j)/2:
            D[0,j[p]] = a[p]
            D[j[p],0] = a[-p-1]
            p = p + 1
        
        
        self.matrix = D

    def __matmul__(self, other):
        return self.matrix @ other
    


class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.grid = grid
        
        ##define needed variables
        size = convergence_order + derivative_order  + 2
        
        j = np.arange(int(-(size+1)/2+1),int((size+1)/2))
        b = np.zeros(size,dtype = int)
        b[self.derivative_order] = 1
        
        
        grid = self.grid.values
        length = self.grid.length
        N = self.grid.N
        
        ##create x matrix
        n = 0
        s = 0
        x = np.array([list(np.zeros(size)) for x in range(N)])
        while n < N-max(j):
            while s < size:
                x[n][s]=grid[n+j[s]]
                s = s+1
            s = 0
            n = n+1
        s = 0
        r = 0
        while n<N:
            while n+j[s]<N:
                x[n][s] = grid[n+j[s]]
                s = s+1
            while s<size:
                x[n][s] = grid[r]
                s = s+1 
                r = r+1
            s = 0
            r = 0
            n = n+1

        ##create h matrix
        h = np.array([list(np.zeros(size)) for x in range(N)])
        i = 0
        m = 0
        while i<N:
            while m<(size-1)/2:
                h[i][m] = abs(x[i][m+1]-x[i][m])
                m = m+1
            h[i][m] = 0
            m = m+1
            while m<size:
                h[i][m] = abs(x[i][m]-x[i][m-1])
                m = m+1
            m = 0
            i = i+1
        for u in h:
            v = 0
            while v<size:
                if u[v] == grid[-1]:
                    u[v] = length-grid[-1]
                v = v+1

        jn = np.array([list(np.zeros(size)) for x in range(N)])
        i = 0
        m = 0
        while i<N:
            while m<(size-1)/2:
                jn[i][m] = -sum(h[i][m:int((size-1)/2)])
                m = m+1
            h[i][m] = 0
            m = m+1
            while m<size:
                jn[i][m] = sum(h[i][int((size-1)/2):m+1])
                m = m+1
            m = 0
            i = i+1

        
        
        ##create a matrix using individual s matrix
        a = [np.zeros(size).tolist() for x in range(N)]
        n = 0
        while n < N:
            S = []
            i = 0
            while i<size:
                S.append(list(((jn[n])**i)/factorial(i)))
                i = i+1
            a[n] = np.linalg.solve(S,b)
           # print(a[n])
            n = n+1

        

        ##transpose a
        counter = 0
        ax = []
        while counter <size:  
            ax.append([sub[counter] for sub in a])
            counter = counter + 1
        
        print(ax)
        c = 0
        while c<(size-1)/2:
            if j[c]!=0:
                ax[c] = ax[c][abs(j[c]):N-abs(j[c])+5] + [0 for x in range(abs(j[c]))]
                c = c+1
            else:
                c = c+1
        print(np.array(ax))
        
        ##create sparse matrix
        D = sparse.diags(ax, offsets=j,shape = [N,N])
        D = D.tocsr()

        ##impute corner cases
        p = 0
        q = 0
        r = 0
        while p < size:
            while q < size:
                D[p,j[q]] = a[p][r]
                D[-(p+1),-j[q]-1] = a[-p-1][-r-1]
                q = q+1
                r = r + 1
            p = p + 1
            q = p
            r = 0 

        ##assign attribute
        self.matrix = D


    def __matmul__(self, other):
        return self.matrix @ other
    
    
class CenteredFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix

