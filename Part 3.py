import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat

import os 
try:
    os.chdir('Datenanalysis_II_project')
except :
    pass

def print_time_needed(time_needed):
    if time_needed >= 60:
        time_needed /= 60
        print(f'done in {np.round(time_needed, 2)} min')
    else:
        print(f'done in {np.round(time_needed, 2)} s')

def part_3(N_simulations, use_accept_reject=True, generate_plot=True):
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

    # tau_pi_plus = 2.6033e-8 # s
    # tau_pi_zero = 8.43e-17 # s
    
    # m_pi_plus = 139.57039 # Mev
    # m_pi_zero = 134.9768  # Mev

    tau_pi = 2.6033e-8 # s    
    m_pi = 139.57039 # Mev

    # -----------------------------------------------
    velocity = lambda l, tau: l/(np.sqrt((tau**2) + ((l**2)/c**2)))
    momentum = lambda l, m, tau: (m*l)/(c*tau)
    energy = lambda m, p: np.sqrt(m**2 + p**2)
    beta = lambda p, E: p/E
    gamma = lambda E, m: E/m

    # K+
    v_k = velocity(l_k, tau_k)
    b_k = v_k/c
    g_k = 1/np.sqrt(1-(b_k**2))

    # # pi+
    # v_pi_plus = velocity(l_pi, tau_pi_plus)
    # b_pi_plus = v_pi_plus/c
    # g_pi_plus = 1/np.sqrt(1-(b_pi_plus**2))
    # p_pi_plus = g_pi_plus * m_pi_plus * v_pi_plus
    # E_pi_plus = np.sqrt(m_pi_plus**2 + p_pi_plus**2)

    # # pi0
    # v_pi_zero = velocity(l_pi, tau_pi_zero)
    # b_pi_zero = v_pi_zero/c
    # g_pi_zero = 1/np.sqrt(1-(b_pi_zero**2))
    # p_pi_zero = g_pi_zero * m_pi_zero * v_pi_zero
    # E_pi_zero = np.sqrt(m_pi_plus**2 + p_pi_plus**2)

    # pi
    v_pi = velocity(l_pi, tau_pi)
    b_pi = v_pi/c
    g_pi = 1/np.sqrt(1-(b_pi**2))
    p_pi = g_pi * m_pi * v_pi_
    E_pi = np.sqrt(m_pi**2 + p_pi**2)

    # -----------------------------------------------

    boost = np.array([
        [    g_k,   0,   0, g_k*b_k], 
        [      0,   1,   0,       0], 
        [      0,   0,   1,       0], 
        [g_k*b_k,   0,   0,     g_k]
    ])

    # run simulation --------------------------------
    t1 = time.time()

    print(f'simulating {N_simulations} decays...')

    pos_data = np.zeros(N_simulations)
    angle_data = np.zeros(N_simulations)
    angle_lab_frame_plus = np.zeros(N_simulations)
    angle_lab_frame_zero = np.zeros(N_simulations)

    for i in range(N_simulations):
        angle_data[i] = np.random.uniform(0, 2*np.pi)
        pos_data[i] = np.random.exponential(scale=l_k)

        # p_pi_plus_vec = np.array([
        #     E_pi_plus, 
        #     0, 
        #     p_pi_plus*np.sin(angle_data[i]), 
        #     p_pi_plus*np.cos(angle_data[i])
        # ])
        # p_pi_zero_vec = np.array([
        #     E_pi_zero, 
        #     0, 
        #     -p_pi_zero*np.sin(angle_data[i]), 
        #     -p_pi_zero*np.cos(angle_data[i])
        # ])

        p_pi_plus_vec = np.array([
            E_pi, 
            0, 
            p_pi*np.sin(angle_data[i]), 
            p_pi*np.cos(angle_data[i])
        ])
        p_pi_zero_vec = np.array([
            E_pi, 
            0, 
            -p_pi*np.sin(angle_data[i]), 
            -p_pi*np.cos(angle_data[i])
        ])

        p_pi_plus_boosted = boost.dot(p_pi_plus_vec)
        p_pi_zero_boosted = boost.dot(p_pi_zero_vec)

        # store data
        angle_lab_frame_plus[i] = abs(np.arctan(p_pi_plus_boosted[2]/p_pi_plus_boosted[3]))
        angle_lab_frame_zero[i] = abs(np.arctan(p_pi_zero_boosted[2]/p_pi_zero_boosted[3]))

    df = pd.DataFrame({
        'decay length K+' : pos_data, 
        'angle pi+' : angle_lab_frame_plus, 
        'angle pi0' : angle_lab_frame_zero,
    })
    df.to_csv('data/simulation_data.csv',index=False)

    t2 = time.time()
    print_time_needed(t2 - t1)

    # optimize detector posititon -------------------
    t1 = time.time()
    print('optimizing detector posititon...')

    # histogram method
    z_max = 800 # the upper bound for the positiion of the 2nd detector in meter
    resolution = 1 # m
    histogram_array, lower_ends, upper_ends = make_histogram_array('data/simulation_data.csv', resolution, z_max)
    optimal_z = int(np.median(np.where(histogram_array == np.max(histogram_array))))
    optimal_z_histogram = optimal_z * resolution
    print('optimal detector position:', optimal_z_histogram, 'm') 

    t2 = time.time()
    print_time_needed(t2 - t1)

    # accept reject method
    if use_accept_reject:
        t1 = time.time()
        print('optimizing detector posititon with accept reject method ...')
        
        data_df = pd.read_csv('data/simulation_data.csv')
        z_pos = np.linspace(150, 350, 400)
        scores = [accept_reject(data_df, z) for z in z_pos]
        optimal_z_idx = int(np.median(np.where(scores == np.max(scores))))
        optimal_z_accept_reject = np.round(z_pos[optimal_z_idx],2)
        print('optimal detector position:', optimal_z_accept_reject, 'm')

        t2 = time.time()
        print_time_needed(t2 - t1)

        print('-----------------------------------------------------------')
    else:
        print('--------------------------------')

    # plot ------------------------------------------
    if generate_plot:
        if resolution == 1: unit = '[m]'
        elif resolution == 0.1: unit = '[dm]'
        elif resolution == 0.01: unit = '[cm]'
        elif resolution == 0.001: unit = '[mm]'
        else: unit = f'[m/{1/resolution}]'
        plt.figure(figsize=(12,4))
        plt.xlabel(f'Detector position {unit}')
        plt.ylabel('Number of Dectections')
        markersize = 3
        if use_accept_reject:
            plt.title('Comparison of the Histogram Method and the Accept Reject Method')
            plt.plot(histogram_array,'o', markersize=markersize, label='Histogram Method')
            plt.plot(z_pos,scores,'x', markersize=markersize, label='Accept Reject Method')
            plt.legend()
            plt.savefig('plots/Comparison_of_the_Histogram_Method_and_the_Accept_Reject_Method.pdf')
        else:
            plt.title('Number of detections at different detector positions')
            plt.plot(histogram_array,'o', markersize=markersize)
            plt.savefig('plots/Number_of_Detections_at_different_detector_positions.pdf')
        plt.show()
    
    return optimal_z_histogram

part_3(N_simulations=1000, use_accept_reject=False)

# N_experiments = 50
# T1 = time.time()
# optimal_pos = np.array([part_3(N_simulations=500000, use_accept_reject=False, generate_plot=False) for _ in range(N_experiments)])
# T2 = time.time()
# np.savetxt('data/optimal_pos.txt', optimal_pos)
# optimal_pos_mean = np.mean(optimal_pos)
# optimal_pos_uncertainty = np.std(optimal_pos) / np.sqrt(len(optimal_pos))
# print('optimal detector position: ${:L}$ m'.format(ufloat(optimal_pos_mean, optimal_pos_uncertainty)))
# print_time_needed(T2 - T1)