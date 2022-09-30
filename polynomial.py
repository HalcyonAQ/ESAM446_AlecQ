#!/usr/bin/env python
# coding: utf-8

# In[381]:

pip install numpy
from numpy import numpy as np
class polynomial:
    def __init__(self,coefficients):
        self.coefficients = coefficients
    @staticmethod
    def from_string(str):
        raw = str.split(" ")
        i=0
        while i<len(raw):
            if raw[i][0] == "x":
                raw[i] = "1*"+raw[i]
                i = i+1
            elif raw[i].find("-x") == 0:
                raw[i] = "-1*"+raw[i]
                i = i+1
            elif raw[i] == "+":
                raw[i+1] = raw[i+1]
                del(raw[i])
            elif raw[i] == "-":
                raw[i+1] = "-"+raw[i+1]
                del(raw[i]) 
            else:
                i = i+1
        i = 0
        while i<len(raw):
            if raw[i].find("^")==-1:
                if raw[i].find("x") >=0:
                    raw[i] += "^1"
                else:
                    raw[i] +="*x^0"
            i = i+1
        orders = []
        i=0
        while i<len(raw):
            orders.append(int(raw[i].split("^")[1])) 
            i = i+1
        coeffs = np.zeros(max(orders)+1,dtype=int)
        for j in raw:
            coeffs[int(j.split("^")[1])] = int(j.split("*")[0])  
        return polynomial(coeffs)
    def __repr__(self):
        string = ""
        i = 0
        while i<len(self.coefficients):
            if self.coefficients[i] == 0:
                string = string
            else:
                if i!=0: 
                    if i == 1:
                        if repr(self.coefficients[i])[0]!="-":
                            if string != "":
                                if self.coefficients[i] == 1:
                                    string+="+ "+"x"+" "
                                else:
                                    string+="+ "+repr(self.coefficients[i])+"*x"+" "
                            else:
                                if self.coefficients[i] == 1:
                                    string+= "x"+" "
                                else:
                                    string+=repr(self.coefficients[i])+"*x"+" "
                        else:
                            string += "- "+repr(self.coefficients[i]).split("-")[1]+"*x"+ " "
                    else:
                        if repr(self.coefficients[i])[0]!="-":
                            if string != "":
                                if self.coefficients[i] == 1:
                                    string+="+ "+"x^"+repr(i)+" "
                                else:
                                    string+="+ "+repr(self.coefficients[i])+"*x^"+repr(i)+" "
                            else:
                                if self.coefficients[i] == 1:
                                    string+= "x^"+repr(i)+" "
                                else:
                                    string+=repr(self.coefficients[i])+"*x^"+repr(i)+" "
                        else:
                            string += "- "+repr(self.coefficients[i]).split("-")[1]+"*x^"+repr(i) + " "
                else:
                    string += repr(self.coefficients[i]) + " "
            i += 1
        return string
    def __eq__(self, other):
        return np.array_equal(self.coefficients,other.coefficients)
    def __add__(self,other):
        coeffs = np.zeros(max(len(self.coefficients),len(other.coefficients))+1,dtype=int)
        i = 0
        while i<len(self.coefficients):
            coeffs[i] += self.coefficients[i]
            i += 1
        i = 0
        while i<len(other.coefficients):
            coeffs[i] += other.coefficients[i]
            i += 1
        j = len(coeffs)-1
        while j>0:
            if coeffs[j] == 0:
                coeffs = np.delete(coeffs,j)
                j = len(coeffs)-1
            else:
                break
        return polynomial(coeffs)
    def __sub__(self,other):
        coeffs = np.zeros(max(len(self.coefficients),len(other.coefficients))+1,dtype=int)
        i = 0
        while i<len(self.coefficients):
            coeffs[i] += self.coefficients[i]
            i += 1
        i = 0
        while i<len(other.coefficients):
            coeffs[i] -= other.coefficients[i]
            i += 1
        j = len(coeffs)-1
        while j>0:
            if coeffs[j] == 0:
                coeffs = np.delete(coeffs,j)
                j = len(coeffs)-1
            else:
                break
        return polynomial(coeffs)
    def __mul__(self,other):
        coeffs = np.zeros(len(self.coefficients)+len(other.coefficients)+1,dtype=int)
        i=0
        j=0
        while i<len(self.coefficients):
            while j<len(other.coefficients):
                coeffs[i+j] += self.coefficients[i]*other.coefficients[j]
                j=j+1
            j=0
            i=i+1
        j = len(coeffs)-1
        while j>0:
            if coeffs[j] == 0:
                coeffs = np.delete(coeffs,j)
                j = len(coeffs)-1
            else:
                break    
        return polynomial(coeffs)
    def __truediv__(self,other):
        return RationalPolynomial(self,other)

import sympy

class RationalPolynomial:
    def __init__(self,numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self._reduce(self)
    @staticmethod
    def _reduce(self):
        n = sympy.sympify(repr(self.numerator))
        d = sympy.sympify(repr(self.denominator))
        gcd = sympy.gcd(n,d)
        [n,r1] = sympy.div(n,gcd,domain = "QQ")
        [d,r2] = sympy.div(d,gcd,domain = "QQ")
        self.numerator = polynomial.from_string(repr(n).replace("**","^"))
        self.denominator = polynomial.from_string(repr(d).replace("**","^"))
    def from_string(str):
        n = str.split("/")[0]
        d = str.split("/")[1]
        return RationalPolynomial(polynomial.from_string(n[1:-1]),polynomial.from_string(d[1:-1]))
    def __repr__(self):
        return "("+repr(self.numerator)+")" + "/" + "("+repr(self.denominator)+")"
    def __add__(self,other):
        return RationalPolynomial(self.numerator+other.numerator,self.denominator+other.denominator)
    def __sub__(self,other):
        return RationalPolynomial(self.numerator-other.numerator,self.denominator-other.denominator)
    def __mul__(self,other):
        return RationalPolynomial(self.numerator*other.numerator,self.denominator*other.denominator)
    def __truediv__(self,other):
        return RationalPolynomial(self.numerator*other.denominator,self.denominator*other.numerator)
    def __eq__(self,other):
        return (self.numerator * other.denominator) == (self.denominator * other.numerator)


# In[ ]:




