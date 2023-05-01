import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from uncertainties import ufloat
from uncertainties import unumpy as unp

import os 
try:
    os.chdir('Datenanalysis_II_project')
except :
    pass


# Functions -------------------------------------
def T(x, l_k):
    def P(x, l_p):
        return 1/l_p * np.exp(- 1/l_p * x)

    def K(x, l_k):
        return 1/l_k * np.exp(- 1/l_k * x)
    
    l_p = 4188 # m
    return 0.84 * P(x,l_p) + 0.16 * K(x,l_k)

def negative_log_of_likelihood_of_data(data, l_k):
    '''
        returns the log of the likelihood because the likelihood gets lower then the computer precision.
    '''
    log_of_likelihood = 0
    for x in data:
        log_of_likelihood += np.log(T(x, l_k))
    return -log_of_likelihood

def function_to_minimize(variables, data):
    '''
        We want to maximize the likelihood, but can only minimize a function.
        => minimize -likelihood
    '''
    l_k = variables[0]
    return negative_log_of_likelihood_of_data(data, l_k)

# load data -------------------------------------
data = np.loadtxt('data/dec_lengths.txt')

# fit -------------------------------------------
counts, bins, _ = plt.hist(data, bins=500, density=True, label='Data')
xdata = bins[:-1]

res = minimize(function_to_minimize, x0=[500], args=(data), method='Powell')
mean_decay_lenght_Kaon = res.x[0]
print(f'mean decay lenght Kaon {mean_decay_lenght_Kaon} m')

# uncertainties ---------------------------------
# we want to find the amount l_k has to be varied in order to change the neg log likelihood by 0.5 units

# get neg log likelihood curve
Xdata = np.linspace(550,574,8000)
Ydata = np.loadtxt('data/neg log likelihood Ydata.txt')

### This can be used to generate Ydata. But it can become computational expencive with more Xdata ###
# Ydata = [negative_log_of_likelihood_of_data(data, l_k) - res.fun - 0.5 for l_k in Xdata]          #
# np.savetxt('data/neg log likelihood Ydata.txt', Ydata)                                            #
#####################################################################################################

# fit
fit_func = lambda x, a, b, c: a*x**2 + b*x + c
popt, pcov = curve_fit(fit_func, Xdata, Ydata)
perr = np.sqrt(np.diag(pcov))
a = ufloat(popt[0], perr[0])
b = ufloat(popt[1], perr[1])
c = ufloat(popt[2], perr[2])

print('a: {:L}'.format(ufloat(popt[0], perr[0])))
print('b: {:L}'.format(ufloat(popt[1], perr[1])))
print('c: {:L}'.format(ufloat(popt[2], perr[2])))

# find root
root = (-b + unp.sqrt(b**2 - 4*a*c)) / (2*a)
uncertainty = np.abs(root - mean_decay_lenght_Kaon)
mean_decay_lenght_Kaon_uncertainty = uncertainty.n + uncertainty.s

print('uncertainty: {:L}'.format(uncertainty))
print('mean decay lenght Kaon uncertainty:', mean_decay_lenght_Kaon_uncertainty)

# mean life time of Kaon ------------------------
l_k = ufloat(mean_decay_lenght_Kaon,mean_decay_lenght_Kaon_uncertainty) # m
l_p = ufloat(4188,0) # m
m_k = ufloat(493.677, 0.016) # MeV
m_p = ufloat(139.57039, 0.00018) # MeV
tau_p = ufloat(2.6033e-8, 0.0005e-8) # s

tau_k = l_k/l_p * tau_p * m_k/m_p
print(r'measured mean decay lenght $l_{K^+} =', '{:L}$'.format(l_k), 'm')
print(r'measured mean life time $\tau_{K^+, \, measured} =', '{:L}$'.format(tau_k), 's')
print(r'literature mean life time $\tau_{K^+,  \, literature} = \left(1.2380 \pm 0.0020\right) \times 10^{-8}$ s')

# plot ------------------------------------------
plt.plot(xdata,T(xdata, res.x[0]), label='Fit')
plt.xlabel('Decay Length [m]')
plt.ylabel('Number of Particles (normalized)')
plt.title(r'Particle decay length of mixed $K^+$ and $\pi^+$ beam')
plt.legend()
plt.savefig('plots/decay_length_of_mixed_beam.pdf')
plt.show()