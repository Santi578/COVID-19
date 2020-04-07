# -*- coding: utf-8 -*-
# 
# Graphics of the solutions equations for the SIR, COMPARTMENTAL MODEL.
# ------------------------------------------------------------------------------
# Written by : Michael Heredia Pérez
# Date       : Apr/2020
# e-mail     : mherediap@unal.edu.co
# Universidad Nacional de Colombia, Manizales campus.
# ------------------------------------------------------------------------------

# LIBRARIES.
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint      # ODF Solver.

# BEGINING DATA
N = 10000           # Total poluation (Colombia case)
Io, Ro = 1, 0       # Day cero : The first infected is anounced (1) and we have 
                    # no recovered people.
So = N - Io - Ro    # Everyone is susceptible to get infected, but an infected 
                    # does not count nor who is already recovered.
beta  = 0.2         # Effective contact rate of the disease: an infected 
                    # individual comes into contact with betta*N other 
                    # individuals per unit time. 
gamma = 1./10       # Mean recovery rate. Mean period of time during which an 
                    # infected individual can pass it on.
tt = np.linspace(0, 180, 180)   # Time window, counting it in days.

# SIR COMPARTMENTAL MODEL ODEs.
def SIR_ODE(y, t, N, beta, gamma):
    '''
    The equations for these model take into considerations variations in 3 
    compartments: Susceptibles S(t), Infectious I(t) and Recovereds R(t) 
    including dead peaople, as:

    S(t+Dt) = S(t) - beta*S*I*Dt/N            --> S'(t) = - beta*S*I/N
    I(t+Dt) = I(t) + beta*S*I*Dt/N - gamma*I  --> I'(t) =   beta*S*I/N - gamma*I
    R(t+Dt) = R(t) + gamma*I                  --> R'(t) =   gamma*I
    
    INPUTS : <> y     : conditions (initials)
             <> t     : time
             <> N     : total population for the case.
             <> beta  : contact rate.
             <> gamma : mean recovery rate.  
    '''

    S, I, R = y                 
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# %% MAIN

initial_conditions = So, Io, Ro

# Solving the SIR_ODE system
Sol = odeint(SIR_ODE, initial_conditions, tt, args = (N, beta, gamma))
S, I, R = Sol.T

# Making the plot.
plt.figure()
plt.plot(tt, S/1000, 'b', alpha = 0.5, lw = 2, label = 'Susceptible')
plt.plot(tt, I/1000, 'r', alpha = 0.5, lw = 2, label = 'Infected')
plt.plot(tt, R/1000, 'g', alpha = 0.5, lw = 2, label = 'Recovered')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Time [days]')
plt.ylabel('Numer [1000s]')
plt.show()