import numpy as np
import matplotlib.pyplot as plt

data_size = 10000

pi = np.pi
mean_decay_length = 537
kaon_lifetime = 1.2385*(10**-8)
pion_lifetime = 2.6033*(10**-8)
kaon_mass = 8.837*(10**-28)
pi_plus_mass = 2.498*(10**-28)
pi_zero_mass = 2.416*(10**-28)
c = 299792458

decay_vertex_positions = np.random.exponential(scale=mean_decay_length, size=data_size)
decay_angles = np.random.uniform(low=0, high=2*pi, size=data_size)

def velocity(d, T):
    numerator = d
    denominator = np.sqrt((T**2) + ((d**2)/c**2))
    return numerator/denominator

kaon_velocity = velocity(mean_decay_length, kaon_lifetime)
pi_velocity = velocity(4188, pion_lifetime)
print(pi_velocity)
print(pi_velocity/c)
B = kaon_velocity/c
G = 1/np.sqrt(1-(B**2))

pi_B = pi_velocity/c
pi_G = 1/np.sqrt(1-(pi_B**2))


lorentz_matrix = np.array([
    [  G,   0,   0, G*B], 
    [  0,   1,   0,   0], 
    [  0,   0,   1,   0], 
    [G*B,   0,   0,   G]
])
    
def detections(detector_position):
    sumk = 0
    for i in range(len(decay_vertex_positions)):
        p_plus = pi_G* pi_plus_mass * pi_velocity
        p_zero = pi_G* pi_zero_mass * pi_velocity
        if detector_position < decay_vertex_positions[i]:
            continue
        else:
            max_angle = np.arctan(2/(detector_position -decay_vertex_positions[i]))
            pi_plus_four_vector = [
                np.sqrt((pi_plus_mass**2)+(p_plus**2)), 
                0, 
                p_plus * np.sin(decay_angles[i]), 
                p_plus * np.cos(decay_angles[i])
            ]
            pi_zero_four_vector = [
                np.sqrt((pi_zero_mass**2)+(p_zero**2)), 
                0, 
                -p_zero * np.sin(decay_angles[i]), 
                -p_zero * np.cos(decay_angles[i])
            ]
            
            boosted_pi_plus_four_vector = lorentz_matrix.dot(pi_plus_four_vector)
            boosted_pi_zero_four_vector = lorentz_matrix.dot(pi_zero_four_vector)
            
            pi_plus_angle = np.arccos(boosted_pi_plus_four_vector[-1] / np.sqrt(boosted_pi_plus_four_vector[-1]**2 + boosted_pi_plus_four_vector[-2]**2 + boosted_pi_plus_four_vector[-3]**2))
            pi_zero_angle = np.arccos(boosted_pi_zero_four_vector[-1] / np.sqrt(boosted_pi_zero_four_vector[-1]**2 + boosted_pi_zero_four_vector[-2]**2 + boosted_pi_zero_four_vector[-3]**2))

            if pi_plus_angle < max_angle and pi_zero_angle < max_angle:
                sumk += 1
    return sumk

detector_positions = np.linspace(175, 274, 100)
number_of_successes = []
for detector_position in detector_positions:
    number_of_successes.append(detections(detector_position))

plt.plot(detector_positions, number_of_successes, 'o', markersize=1, color = 'black')
plt.title('The Number of Detections as a Function of the Detector Position')
plt.xlabel('Detector Position in meters')
plt.ylabel('Detections')
plt.savefig('results')
plt.show()