# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:41:17 2021

@author: phili
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint

# conversion units so all quantities are dimensionless

l = 1.50*10**11 # m

m = 1.99*10**30 # kg

t = 5.0425*10**6 # s

# getting started, we consider the two masses to be the sun and earth. 

G = 6.67*10**-11*(m*t**2/l**3)

ms = 1.99*10**30/m

me = 5.97*10**24/m

a = 1.50*10**11/l

e = 0.0167086

# numerical integral for t(r) - not helpful at the moment 
def integrand(r, a, e):
    return (1/(np.sqrt(r - r**2/(2*a) - (a*(1-e**2))/2)))*r

# numerical integral for t(r)

#I = quad(integrand, 3, 5, args = (a, e))

# numerically integrating the second order differential equation for r(t)
def orb(y, t, a, e):
    r, v = y
    dvdt = [v, G*(ms+me)*(a*(1-e**2)/r**3 - 1/r**2)]
    return dvdt

# initial conditions

r0 = [a*(1-e**2)/(1+e), 0.0]

# range of t to plot over

t = np.linspace(0, 5000, 50001)

# numerically integrate and store the solution for r(t) and v(t) 
# in the vector sol

sol = odeint(orb, r0, t, args=(a, e))

# plotting r(t) and v(t) vs. t

plt.plot(t, sol[:, 0], 'b', label='r(t)')
plt.plot(t, sol[:, 1], 'g', label='v(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

# phase space plot v(t) vs. r(t)

plt.plot(sol[:,0], sol[:,1])
plt.xlabel('r(t)')
plt.ylabel('v(t)')
plt.grid()
plt.show()

# finding indices where v approximately equals zero to make 
# line of section plot

ivzero = []

for i in range(0, len(sol[:,1])-1):
    if (sol[i,1]/sol[i+1,1])<0:
       ivzero = np.append(ivzero, i)
       
# converting indices to integers
     
ivzero = ivzero.astype(int)
     
plt.scatter(sol[ivzero,0], 0*sol[ivzero,1], s= 1 )


