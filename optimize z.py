import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_test_data(N):
    decay_length = np.random.exponential(scale=562, size=N)
    angle_1 = np.random.uniform(0, np.pi/4, size=N)
    angle_2 = np.random.uniform(0, np.pi/4, size=N)

    test_data_df = pd.DataFrame({
        'decay lenght K+' : decay_length,
        'angle pi+' : angle_1,
        'angle pi0' : angle_2
    })

    test_data_df.to_csv('test_data.csv',index=False)

def make_histogram_array(data_file, resolution, z_max):
    half_detector_height = 2 * 1/resolution 
    histogram_array = np.zeros(int(z_max * 1/resolution))
    data_df = pd.read_csv(data_file)
    lower_ends = np.zeros(len(data_df))
    upper_ends = np.zeros(len(data_df))

    # data_df = data_df.reset_index()  # make sure indexes pair with number of rows
    for i, row in data_df.iterrows():
        dK = row['decay lenght K+']
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

def function_to_minimize(z, upper_and_lower_ends):
    lower_ends = upper_and_lower_ends[0]
    upper_ends = upper_and_lower_ends[1]

    score = 0
    for lower_end, upper_end in zip(lower_ends, upper_ends):
        if lower_end <= z and z <= upper_end:
            score += 1

    return -score

# generate_test_data(1000)

z_max = 5000 # the upper bound for the positiion of the 2nd detector in meter
resolution = 0.001 # m
histogram_array, lower_ends, upper_ends = make_histogram_array('test_data.csv', resolution, z_max)
optimal_z = int(np.median(np.where(histogram_array == np.max(histogram_array)))) 
print('optimal z:', optimal_z * resolution, 'm')

# res = minimize(function_to_minimize, x0=[2038000], args=([lower_ends, upper_ends]), method='Powell')
# print(res)

plt.plot(histogram_array)
plt.scatter(optimal_z,histogram_array[optimal_z],color='red')
# plt.scatter(optimal_z,0,color='red')
# plt.scatter(res.x[0],0,color='green')
plt.show()
    
