import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp

try:
    os.chdir('Datenanalysis_II_project')
except:
    pass

# 2 Determination of the average decay length of the K+ ###########################################
# Functions -------------------------------------
def P(x, l_p):
    return 1/l_p * np.exp(- 1/l_p * x)

def K(x, l_k):
    return 1/l_k * np.exp(- 1/l_k * x)

def T(x, l_k):
    l_p = 4188 # m
    return 0.84 * P(x,l_p) + 0.16 * K(x,l_k)

def likelihood_of_data(data, l_k):
    '''
        returns the log of the likelihood because the likelihood gets lower then the computer precision.
    '''
    log_of_likelihood = 0
    for x in data:
        log_of_likelihood += np.log(T(x, l_k))
    return log_of_likelihood

def function_to_minimize(variables, data):
    '''
        We want to maximize the likelihood, but can only minimize a function.
        => minimize -likelihood
    '''
    l_k = variables[0]
    return -likelihood_of_data(data, l_k)

# load data -------------------------------------
data = np.loadtxt('dec_lengths.txt')

# fit -------------------------------------------
res = minimize(function_to_minimize, x0=[500], args=(data), method='Powell')
mean_decay_lenght_Kaon = res.x[0]
print(f'mean decay lenght Kaon {mean_decay_lenght_Kaon} m')

# uncertainties ---------------------------------
# we want to find the amount l_k has to be varied in order to change the neg log likelihood by 0.5 units

# get neg log likelihood curve
Xdata = np.linspace(550,574,8000)
# Ydata = [negative_log_of_likelihood_of_data(data, l_k) - res.fun - 0.5 for l_k in Xdata] 
# np.savetxt('neg log likelihood Ydata.txt', Ydata)
Ydata = np.loadtxt('neg log likelihood Ydata.txt')

# fit
fit_func = lambda x, a, b, c: a*x**2 + b*x + c
popt, pcov = curve_fit(fit_func, Xdata, Ydata)
perr = np.sqrt(np.diag(pcov))
a = ufloat(popt[0], perr[0])
b = ufloat(popt[1], perr[1])
c = ufloat(popt[2], perr[2])
# a = popt[0]
# b = popt[1]
# c = popt[2]

print('a: {:L}'.format(ufloat(popt[0], perr[0])))
print('b: {:L}'.format(ufloat(popt[1], perr[1])))
print('c: {:L}'.format(ufloat(popt[2], perr[2])))

# find root
try:
    root = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
except:
    root = (-b + unp.sqrt(b**2 - 4*a*c)) / (2*a)

uncertainty = np.abs(root - mean_decay_lenght_Kaon)
try:
    print('uncertainty: {:L}'.format(uncertainty))
    mean_decay_lenght_Kaon_uncertainty = uncertainty.n + uncertainty.s
except:
    print('uncertainty:', uncertainty)
    mean_decay_lenght_Kaon_uncertainty = uncertainty

# plot ------------------------------------------
counts, bins, _ = plt.hist(data, bins=500, density=True, label='Data')
xdata = bins[:-1]
plt.plot(xdata,T(xdata, res.x[0]), label='Fit')
plt.xlabel('Decay Length [m]')
plt.ylabel('Number of Particles (normalized)')
plt.title(r'Particle decay length of mixed $K^+$ and $\pi^+$ beam')
plt.legend()
plt.savefig('decay_length_of_mixed_beam.pdf')
# plt.show()

# mean life time of Kaon ------------------------
l_k = ufloat(mean_decay_lenght_Kaon,mean_decay_lenght_Kaon_uncertainty) # m
l_p = ufloat(4188,0) # m
m_k = ufloat(493.677, 0.016) # MeV
m_p = ufloat(139.57039, 0.00018) # MeV
tau_p = ufloat(2.6033e-8, 0.0005e-8) # s

tau_k = l_k/l_p * tau_p * m_k/m_p
print(r'measured mean life time $\tau_{K^+} =', '{:L}$'.format(tau_k), 's')
print(r'literature mean life time $\tau_{K^+} = \left(1.2380 \pm 0.0020\right) \times 10^{-8}$ s')

# 3 Infinitely narrow beam along the z axis #######################################################

c = 299792458 # m/s
l_k = l_k.n # measured mean decay length of Kaon
l_pi = l_p.n # mean decay length of Pion
tau = tau_p.n # s
mass_kaon = m_k.n # Mev
mass_pi = m_p.n # Mev

p_pi = mass_pi*l_pi/(c*tau) # three momentum of the pion in the Kaon rest frame
E_pion = np.sqrt(mass_pi**2 + p_pi**2)
gamma = E_pion/mass_pi
beta = p_pi/E_pion

boost = np.array([
    [     gamma, 0, 0, beta*gamma],
    [         0, 1, 0,          0],
    [         0, 0, 1,          0],
    [beta*gamma, 0, 0,      gamma]
    ])

decay_pos = []
angle_1 = []
angle_2 = []
for _ in range(10000):
    angle = np.random.uniform(0, 2*np.pi)
    decay_position = np.random.exponential(scale=1/l_k)

    p_pi_plus = np.array([E_pion,0,p_pi*np.cos(angle),p_pi*np.sin(angle)])
    p_pi_neutral = np.array([E_pion,0,-p_pi*np.cos(angle),-p_pi*np.sin(angle)])

    p_pi_plus_boosted = np.dot(boost,p_pi_plus)
    p_pi_neutral_boosted = np.dot(boost,p_pi_neutral)

    angle_lab_frame_plus = abs(np.arctan(p_pi_plus_boosted[2]/p_pi_plus_boosted[3]))
    angle_lab_frame_neutral = abs(np.arctan(p_pi_neutral_boosted[2]/p_pi_neutral_boosted[3]))

    decay_pos.append(decay_position)
    angle_1.append(angle_lab_frame_plus)
    angle_2.append(angle_lab_frame_neutral)

print(np.rad2deg(np.max(angle_1)),np.rad2deg(np.min(angle_1)))
print(np.rad2deg(np.max(angle_2)),np.rad2deg(np.min(angle_2)))

plt.figure()
plt.hist(np.rad2deg(angle_1),bins=500,label='a1')
plt.hist(np.rad2deg(angle_2),bins=500,label='a2')
plt.show()

def make_histogram_array(data_file, resolution, z_max):
    half_detector_height = 2 * 1/resolution 
    histogram_array = np.zeros(int(z_max * 1/resolution))
    data_df = pd.read_csv(data_file)
    lower_ends = np.zeros(len(data_df))
    upper_ends = np.zeros(len(data_df))

    # for i, row in data_df.iterrows():
    #     dK = row['decay lenght K+']
    #     a1 = row['angle pi+']
    #     a2 = row['angle pi0']

    plt.figure()
    plt.hist(data_df['angle pi+'])
    plt.hist(data_df['angle pi0'])


    for i, (dK, a1, a2) in enumerate(zip(decay_pos, angle_1, angle_2)):

        # distance (on the z axis) when paricle leaves detectable range (on the y axis)
        #       /|
        #      / | 2m
        # ____/__|         2m/z = tan(a) --> z = 2m/tan(a)
        #  dK  z

        z1 = half_detector_height/np.tan(a1)
        z2 = half_detector_height/np.tan(a2)

        # the Pions don't exist before the decay of the Kaon
        lower_end = dK * 1/resolution
        lower_ends[i] = lower_end
        lower_idx = int(np.round(lower_end))

        # both Pions need to be detected so stop when the first one leave the detectable range
        upper_end = np.min([dK+z1, dK+z2, z_max]) * 1/resolution
        upper_ends[i] = upper_end
        upper_idx = int(np.round(upper_end)) 

        histogram_array[lower_idx:upper_idx] += 1

    return histogram_array, lower_ends, upper_ends

z_max = 1100 # the upper bound for the positiion of the 2nd detector in meter
resolution = 0.1 # m
histogram_array, lower_ends, upper_ends = make_histogram_array('test_data.csv', resolution, z_max)
optimal_z = int(np.median(np.where(histogram_array == np.max(histogram_array)))) 
print('optimal z:', (optimal_z) * resolution, 'm') 

plt.figure()
plt.plot(histogram_array)
plt.scatter(optimal_z,histogram_array[optimal_z],color='red')
plt.show()
    

# 4 Divergent beam ################################################################################




