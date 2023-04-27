import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# functions -------------------------------------
def make_histogram_array(data_file, resolution, z_max):
    half_detector_height = 2 * 1/resolution 
    histogram_array = np.zeros(int(z_max * 1/resolution))
    data_df = pd.read_csv(data_file)
    lower_ends = np.zeros(len(data_df))
    upper_ends = np.zeros(len(data_df))

    # data_df = data_df.reset_index()  # make sure indexes pair with number of rows
    for i, row in data_df.iterrows():
        dK = row['decay length K+']
        a1 = row['angle pi+']
        a2 = row['angle pi0']

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

def accept_reject(data_df, z):
    half_detector_height = 2 # m
    
    score = 0
    for i, row in data_df.iterrows():
        dK = row['decay length K+']
        if z >= dK:
            angle_plus = row['angle pi+']
            angle_zero = row['angle pi0']

            z_plus = half_detector_height/np.tan(angle_plus)
            z_zero = half_detector_height/np.tan(angle_zero)
            if z <= np.min([dK+z_plus, dK+z_zero]):
                score += 1
    return score

# set up ----------------------------------------
c = 299792458 # m/s
l_k = 562 # m, measured mean decay length of Kaon
l_pi = 4188 # m, mean decay length of Pion
tau_k = 1.2380e-8 # s
tau_pi = 2.6033e-8 # s
# mass_kaon = 8.800595717468e-28 # kg # 493.677 # Mev
# mass_pi = 2.488069277117e-28 # kg # 139.57039 # Mev
mass_kaon = 493.677 # Mev
mass_pi = 139.57039 # Mev

# p_pi = mass_pi*l_pi/(c*tau_pi) # three momentum of the pion in the Kaon rest frame
# E_pi = np.sqrt(mass_pi**2 + p_pi**2)
# gamma = E_pi/mass_pi
# beta = p_pi/E_pi

# -----------------------------------------------
mean_decay_length = l_k
kaon_lifetime = tau_k
pion_lifetime = tau_pi

velocity = lambda l, tau: l/(np.sqrt((tau**2) + ((l**2)/c**2)))

v_k = velocity(l_k, tau_k)
v_pi = velocity(l_pi, tau_pi)
b_k = v_k/c
g_k = 1/np.sqrt(1-(b_k**2))

pi_B = v_pi/c
pi_G = 1/np.sqrt(1-(pi_B**2))

p_pi = pi_G* mass_pi * v_pi
E_pi = np.sqrt(mass_pi**2 + p_pi**2)

# -----------------------------------------------

boost = np.array([
    [    g_k,   0,   0, g_k*b_k], 
    [      0,   1,   0,       0], 
    [      0,   0,   1,       0], 
    [g_k*b_k,   0,   0,     g_k]
])

# run simulation --------------------------------
t1 = time.time()

N_simulations = 1000
print(f'simulating {N_simulations} decays...')

pos_data = np.zeros(N_simulations)
angle_data = np.zeros(N_simulations)
angle_lab_frame_plus = np.zeros(N_simulations)
angle_lab_frame_zero = np.zeros(N_simulations)
z_2_data = np.zeros(N_simulations)

for i in range(N_simulations):
    angle_data[i] = np.random.uniform(0, 2*np.pi)
    pos_data[i] = np.random.exponential(scale=l_k)

    p_pi_plus = np.array([
        E_pi, 
        0, 
        p_pi*np.sin(angle_data[i]), 
        p_pi*np.cos(angle_data[i])
    ])
    p_pi_zero = np.array([
        E_pi, 
        0, 
        -p_pi*np.sin(angle_data[i]), 
        -p_pi*np.cos(angle_data[i])
    ])

    p_pi_plus_boosted = boost.dot(p_pi_plus)
    p_pi_zero_boosted = boost.dot(p_pi_zero)

    angle_plus = abs(np.arctan(p_pi_plus_boosted[2]/p_pi_plus_boosted[3]))
    angle_zero = abs(np.arctan(p_pi_zero_boosted[2]/p_pi_zero_boosted[3]))

    # get z2
    z_plus = 2/np.tan(angle_plus)
    z_zero = 2/np.tan(angle_zero)
    z_2 = np.min([pos_data[i]+z_plus, pos_data[i]+z_zero])

    # store data
    angle_lab_frame_plus[i] = angle_plus
    angle_lab_frame_zero[i] = angle_zero
    z_2_data[i] = z_2

df = pd.DataFrame({
    'decay length K+' : pos_data, 
    'angle pi+' : angle_lab_frame_plus, 
    'angle pi0' : angle_lab_frame_zero,
    'z2' : z_2_data,
})
df.to_csv('simulation_data.csv',index=False)

t2 = time.time()
time_needed = t2 - t1
if time_needed >= 60:
    time_needed /= 60
    print(f'done in {np.round(time_needed, 2)} min')
else:
    print(f'done in {np.round(time_needed, 2)} s')

# optimize detector posititon -------------------
t1 = time.time()
print('optimizing detector posititon...')

# histogram method
z_max = 800 # the upper bound for the positiion of the 2nd detector in meter
resolution = 1 # m
histogram_array, lower_ends, upper_ends = make_histogram_array('simulation_data.csv', resolution, z_max)
optimal_z = int(np.median(np.where(histogram_array == np.max(histogram_array)))) 
print('optimal z:', (optimal_z) * resolution, 'm') 

# # accept reject method
# data_df = pd.read_csv('simulation_data.csv')
# z_pos = np.linspace(150, 350, 600)
# scores = [accept_reject(data_df, z) for z in z_pos]
# optimal_z_idx = int(np.median(np.where(scores == np.max(scores))))
# print('optimize z:', z_pos[optimal_z_idx], 'm')

t2 = time.time()
time_needed = t2 - t1
if time_needed >= 60:
    time_needed /= 60
    print(f'done in {np.round(time_needed, 2)} min')
else:
    print(f'done in {np.round(time_needed, 2)} s')

# plot ------------------------------------------

plt.figure(figsize=(12,4))
plt.xlabel('Detector positiion')
plt.ylabel('Number of Dectections')
markersize = 3
try:
    plt.title('Comparison of the Histogram Method and the Accept Reject Method')
    plt.plot(z_pos,scores,'x', markersize=markersize, label='Accept Reject Method', zorder=10)
    plt.plot(histogram_array,'o', markersize=markersize, label='Histogram Method')
    plt.legend()
except:
    plt.title('Number of Detections at different detector positions')
    plt.plot(histogram_array,'o', markersize=markersize)
plt.show()

