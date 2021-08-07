# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 15:18:35 2021

@author: phili
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# conversion units so all quantities are dimensionless

l = 1.50*10**11 # m

m = 1.99*10**30 # kg

T = 5.0425*10**6 # s

# getting started, we consider the two masses to be the sun and earth. 

G = 6.67*10**-11*(m*T**2/l**3)

m1 = 1.99*10**30/m

m2 = 1.303*10**22/m

# semi major axis

a = 5.90638*10**12/l

# eccentrcity

e = 0.2488

# orbital period

tau = 2*np.pi*np.sqrt(a**3/(G*(m1+m2)))

# numerically integrating the coupled second order differential equations
def orb(y, t):
    r, v, phi, omega = y
    dvdt =  r*omega**2 - G*(m1 + m2)/r**2
    domegadt =  -2*v*omega/r
    return [v, dvdt, omega, domegadt]

# initial conditions

initial = [a*(1-e**2)/(1+e), 0.0, 0.0, np.sqrt(G*(m1+m2))*(1+e)**2/((a*(1-e**2))**(3/2))]

# range of t to plot over in terms of orbital period

t = np.arange(0, 10*tau, tau/100)

# numerically integrate and store the solution for r(t), v(t), phi(t), omega(t)
# in the vector sol

sol = odeint(orb, initial, t)

# r(t) and v(t) vs. t

plt.plot(t/tau, sol[:, 0], 'b', label='r(t)')
plt.plot(t/tau, sol[:, 1], 'g', label='v(t)')
plt.legend(loc='best')
plt.xlabel('t/tau')
plt.grid()
plt.show()

# phi(t) and omega(t) vs. t

plt.plot(t/tau, sol[:, 2], 'b', label='phi(t)')
plt.plot(t/tau, sol[:, 3], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t/tau')
plt.grid()
plt.show()

# phase space plot v(t) vs. r(t)

plt.plot(sol[:,0], sol[:,1])
plt.xlabel('r(t)')
plt.ylabel('v(t)')
plt.grid()
plt.show()

# r(t) vs. phi(t) 

plt.plot(sol[:, 2], sol[:, 0],)
plt.xlabel('phi(t)')
plt.ylabel('r(t)')
plt.grid()
plt.show()

# rescalign phi to reflect periodic angular motion

reducedphi = sol[:,2]

for i in range(0, len(sol[:,2])):
    if i <= 100:
        reducedphi[i] =  reducedphi[i]
    elif 100 < i <= 200:
        reducedphi[i] =  reducedphi[i] - 2*np.pi
    elif 200 < i <= 300:
        reducedphi[i] =  reducedphi[i] - 4*np.pi
    elif 300 < i <= 400:
        reducedphi[i] =  reducedphi[i] - 6*np.pi
    elif 400 < i <= 500:
        reducedphi[i] =  reducedphi[i] - 8*np.pi
    elif 500 < i <= 600:
        reducedphi[i] =  reducedphi[i] - 10*np.pi
    elif 600 < i <= 700:
        reducedphi[i] =  reducedphi[i] - 12*np.pi
    elif 700 < i <= 800:
        reducedphi[i] =  reducedphi[i] - 14*np.pi
    elif 800 < i <= 900:
        reducedphi[i] =  reducedphi[i] - 16*np.pi
    else:
        reducedphi[i] =  reducedphi[i] - 18*np.pi
        
plt.polar(reducedphi, sol[:, 0],)
plt.xlabel('phi(t)')
plt.ylabel('r(t)')
plt.grid()
plt.show()
    

# finding indices where v approximately equals zero to make 
# line of section plot

#ivzero = []

#for i in range(0, len(sol[:,1])-1):
   # if (sol[i,1]/sol[i+1,1])<0:
    #   ivzero = np.append(ivzero, i)
       
# converting indices to integers
     
#ivzero = ivzero.astype(int)
     
#plt.scatter(sol[ivzero,0], 0*sol[ivzero,1], s= 1 )


