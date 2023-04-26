import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import randrange, uniform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import scipy.optimize


#Part 3

#3a)







#3b)





#3c)

c = 299792458 #m/s

l_k = 609.5 #measured mean decay length of kaon
l_pi = 4188 #measured mean decay length of pion

tau = 2.6033*10**(-8) #s
mass_kaon = 493.68 #Mev
mass_pi = 193.57 #Mev
p_pi= mass_pi*l_pi/(c*tau) #three momentum of the pion in the Kaon rest frame

E_pion = np.sqrt(mass_pi**2 + p_pi**2)

gamma = E_pion/mass_pi
beta = p_pi/E_pion



boost = np.array([[gamma, 0, 0, beta*gamma],
         [0, 1, 0, 0]
         [0, 0, 1, 0]
         [beta*gamma, 0, 0, gamma]])

for i in range(100000):
    angle = np.random.uniform(0, 2 * np.pi)
    decay_position = np.random.exponential(scale=1/562)

    p_pi_plus = np.array([E_pion,0,p_pi*np.cos(angle),p_pi*np.sin(angle)])
    p_pi_neutral = np.array([E_pion,0,-p_pi*np.cos(angle),-p_pi*np.sin(angle)])

    p_pi_plus_boosted = np.dot(boost,p_pi_plus)
    p_pi_neutral_boosted = np.dot(boost,p_pi_neutral)

    angle_lab_frame_plus = abs(np.arctan(p_pi_plus_boosted[2]/p_pi_plus_boosted[3]))
    angle_lab_frame_neutral = abs(np.arctan(p_pi_neutral_boosted[2]/p_pi_neutral_boosted[3]))







