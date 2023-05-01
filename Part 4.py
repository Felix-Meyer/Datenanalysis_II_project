import time
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from constants import kaon_lifetime, pion_lifetime, pion_plus_mass, pion_zero_mass, speed_of_light

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

def part_4(num_events, generate_plot=True):
    # Define the mean decay length
    mean_decay_length_kaon = 562
    mean_decay_length_pion = 4188

    t1 = time.time()
    print(f'simulating {num_events} decays...')

    # Generate decay positions, angles, and scattering angles
    kaon_decay_positions = np.random.exponential(scale=mean_decay_length_kaon, size=num_events)
    pion_decay_angles = np.random.uniform(low=0, high=2 * np.pi, size=num_events)
    kaon_scattering_angles = np.random.normal(0, 1 / 1000, num_events)


    def calc_velocity(distance, time):
        time_squared = time ** 2
        velocity_squared = distance ** 2 / (time_squared + distance ** 2 / speed_of_light ** 2)
        velocity = np.sqrt(np.where(time_squared == 0, 0, velocity_squared))
        return velocity


    # Calculate kaon and pion velocities and gammas
    kaon_velocity = calc_velocity(mean_decay_length_kaon, kaon_lifetime)
    pion_velocity = calc_velocity(mean_decay_length_pion, pion_lifetime)

    kaon_beta = kaon_velocity / speed_of_light
    kaon_gamma = 1 / np.sqrt(1 - (kaon_beta ** 2))
    pion_beta = pion_velocity / speed_of_light
    pion_gamma = 1 / np.sqrt(1 - (pion_beta ** 2))


    def lorentz_transform(velocity_y, velocity_z, kaon_velocity):
        gamma = 1 / np.sqrt(1 - (kaon_velocity ** 2 / speed_of_light ** 2))
        lorentz_matrix = np.array([
            [gamma, 0, -gamma * velocity_y / speed_of_light, -gamma * velocity_z / speed_of_light],
            [    0, 1,                                    0,                                    0],
            [-gamma * velocity_y / speed_of_light, 0,   1 + (gamma - 1) * (velocity_y ** 2 / kaon_velocity ** 2), 
            (gamma - 1) * velocity_y * velocity_z / kaon_velocity ** 2],
            [-gamma * velocity_z / speed_of_light, 0, (gamma - 1) * velocity_z * velocity_y / kaon_velocity ** 2,   
            1 + (gamma - 1) * (velocity_z ** 2 / kaon_velocity ** 2)]
        ])
        return lorentz_matrix


    def boost_particle(pion_mass, momentum_x, momentum_y, momentum_z, kaon_velocity, kaon_scattering_angle):
        p = np.sqrt(momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2)
        E = np.sqrt(pion_mass ** 2 + p ** 2)
        pion_four_vector = np.array([E, momentum_x, momentum_y, momentum_z])
        velocity_y = kaon_velocity * np.sin(kaon_scattering_angle)
        velocity_z = kaon_velocity * np.cos(kaon_scattering_angle)
        lorentz_matrix = lorentz_transform(velocity_y, velocity_z, kaon_velocity)
        boosted_pion_four_vector = lorentz_matrix.dot(pion_four_vector)
        return boosted_pion_four_vector


    def boost_particles(num_events, pion_gamma, pion_mass, pion_velocity, pion_decay_angles, kaon_velocity,
                        kaon_scattering_angles):
        boosted_pion_plus_four_vectors = []
        boosted_pion_zero_four_vectors = []

        for i in range(num_events):
            momentum_x = 0
            momentum_y = pion_gamma * pion_mass * pion_velocity * np.sin(pion_decay_angles[i])
            momentum_z = pion_gamma * pion_mass * pion_velocity * np.cos(pion_decay_angles[i])

            # Boost pion+ four-vector
            boosted_pion_plus_four_vector = boost_particle(pion_plus_mass, momentum_x, momentum_y,
                                                        momentum_z, kaon_velocity, kaon_scattering_angles[i])
            boosted_pion_plus_four_vectors.append(boosted_pion_plus_four_vector)

            # Boost pion0 four-vector
            boosted_pion_zero_four_vector = boost_particle(pion_zero_mass, momentum_x, -momentum_y,
                                                        -momentum_z, kaon_velocity, kaon_scattering_angles[i])
            boosted_pion_zero_four_vectors.append(boosted_pion_zero_four_vector)

        return boosted_pion_plus_four_vectors, boosted_pion_zero_four_vectors


    boosted_pion_plus_four_vectors, boosted_pion_zero_four_vectors = boost_particles(num_events, pion_gamma, pion_plus_mass,
                                                                                    pion_velocity, pion_decay_angles,
                                                                                    kaon_velocity, kaon_scattering_angles)

    t2 = time.time()
    print_time_needed(t2 - t1)

    t1 = time.time()
    print('optimizing detector posititon...')


    def detections(detector_positions):
        detections = np.zeros_like(detector_positions)
        for i in range(num_events):
            momentum_pion_plus_z = boosted_pion_plus_four_vectors[i][3]
            velocity_pion_plus_z = momentum_pion_plus_z / pion_plus_mass
            
            momentum_pion_plus_y = boosted_pion_plus_four_vectors[i][2]
            velocity_pion_plus_y = momentum_pion_plus_y / pion_plus_mass

            decay_time_pion_plus = np.abs(detector_positions - kaon_decay_positions[i]) / velocity_pion_plus_z
            distance_pion_plus_y = velocity_pion_plus_y * decay_time_pion_plus

            mask_plus = np.abs(distance_pion_plus_y) < 2

            if np.any(mask_plus):
                momentum_pion_zero = pion_zero_mass
            
                pz_pion_zero = boosted_pion_zero_four_vectors[i][3]
                velocity_pion_zero_z = pz_pion_zero / momentum_pion_zero
            
                py_pion_zero = boosted_pion_zero_four_vectors[i][2]
                velocity_pion_zero_y = py_pion_zero / momentum_pion_zero
                
                decay_time_pion_zero = np.abs(detector_positions - kaon_decay_positions[i]) / velocity_pion_zero_z
                distance_pion_zero_y = velocity_pion_zero_y * decay_time_pion_zero

                mask_zero = np.abs(distance_pion_zero_y) < 2

                detections[mask_plus & mask_zero] += 1

        return detections


    detector_positions = np.linspace(0, 420, 420)
    detection_counts = detections(detector_positions)

    max_index = np.argmax(detection_counts)
    max_count = detection_counts[max_index]
    max_position = detector_positions[max_index]
    
    t2 = time.time()
    print_time_needed(t2 - t1)

    if generate_plot:
        fig, ax = plt.subplots()
        ax.plot(detector_positions, detection_counts, 'o', markersize=1, color='blue')
        ax.set_title('Particle Detections with divergent Kaons')
        ax.set_xlabel('Detector Position [m]')
        ax.set_ylabel('Detections')
        fig.savefig('plots/divergent_kaons.pdf')
        plt.show()

    print(f"Maximum detection count: {max_count}")
    print(f"Detector position with maximum count: {max_position}")
    
    print('--------------------------------')

    return max_position

# part_4(num_events=100000, generate_plot=True)

N_experiments = 50
T1 = time.time()
optimal_pos = np.array([part_4(num_events=500000, generate_plot=False) for _ in range(N_experiments)])
T2 = time.time()
np.savetxt('data/optimal_pos_divergent_kaons.txt', optimal_pos)
optimal_pos_mean = np.mean(optimal_pos)
optimal_pos_uncertainty = np.std(optimal_pos) / np.sqrt(len(optimal_pos))
print('optimal detector position: ${:L}$ m'.format(ufloat(optimal_pos_mean, optimal_pos_uncertainty)))
print_time_needed(T2 - T1)