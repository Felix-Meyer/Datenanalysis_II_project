import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import randrange, uniform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import scipy.optimize

#3a)



#γ=E_K/m_K




dec_lengths = pd.read_csv('dec_lengths.txt',header=None)

dec_lengths.columns = ["decay_lengths"]
dec_lengths_array = np.loadtxt('dec_lengths.txt')



hist, bin_edges = np.histogram(dec_lengths,bins=100)

bin_edges = 3*10**(-3)*bin_edges/np.max(bin_edges)



'''

def exp_K(x,l_K):
    return l_K*np.exp(-l_K*x)

def exp_pi(x):
    return l_pi*np.exp(-l_pi*x)

#def P_gauss(x,sigma,mu):
#    return 0.84/np.sqrt(2*np.pi*sigma**2)np.exp(-(x-mu)**2/(2*sigma**2)) + 0.16/np.sqrt(2*np.pi*sigma**2)np.exp(-(x-mu)**2/(2*sigma**2))


def P(x,l_K):
    return 0.84*l_pi*np.exp(-l_pi*dec_lengths_array[x]**2) + 0.16*np.exp(-l_K*dec_lengths_array[x]**2)


def log_likelihood(l_K):
 #   x, l_K = params
    value = 0
    for i in range(len(dec_lengths_array)):
        value += np.log(P(i,l_K))
    return -value






data = np.loadtxt('dec_lengths.txt')
plt.figure(1)
counts, bins, _ = plt.hist(data,bins=500,density=False)
xdata = bins[:-1]
ydata = counts


def model_func(x,l_k):
    return 0.84*l_pi*np.exp(-x*l_pi) + 0.16*l_k*np.exp(-x*l_k)


def log_likelihood(l_k):
    value = 0
    for i in range(len(ydata)):
        value += np.log(model_func(ydata[i],l_k))
    return -value



#res = scipy.optimize.minimize(log_likelihood,x0=[3],bounds=((1,10)))

#opt_params, params_cov = curve_fit(model_func,xdata,ydata,[6000,1])




#plt.plot(xdata,model_func(np.linspace(0,np.max(xdata),len(xdata)),*opt_params))
#plt.show()

'''

#Part 3
c = 299792458 #m/s

l_k = 609.5 #measured mean decay length of kaon
L_pi = 4188 #measured mean decay length of pion

mass_kaon = 493.68 #Mev
mass_pi_plus = 193.57 #Mev
mass_pi_minus = mass_pi_plus




def gamma(E,m):
    return E/m

def beta(p,E):
    return p/E



#3a)



decay_position = np.random.exponential(scale=1/l_k,size=len(dec_lengths_array))



#3b)

angle_plus_k_rest = np.random.uniform(0,2*np.pi,len(dec_lengths_array))
angle_minus_k_rest = np.random.uniform(0,2*np.pi,len(dec_lengths_array))



#3c)

def lorentz_boost(E,p,m):
    return np.array([[gamma(E,m), 0, 0, beta(p,E)*gamma(E,m)],
                    [0, 1, 0, 0]
                    [0, 0, 1, 0]
                    [beta(p,E)*gamma(E,m), 0, 0, gamma(E,m)]])


p_pi_plus = np.array()


def events(z,E,p,m):
    number = 0
    for i in range(len(dec_lengths_array)):



