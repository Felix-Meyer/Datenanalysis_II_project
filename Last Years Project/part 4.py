import numpy as np
import matplotlib.pyplot as plt

data_size = 10000

pi = np.pi
mean_decay_length = 537
kaon_lifetime = 1.2385*(10**-8)
kaon_mass = 8.837*(10**-28)
pion_lifetime = 2.6033*(10**-8)
pi_plus_mass = 2.498*(10**-28)
pi_zero_mass = 2.416*(10**-28)
c = 299792458

decay_vertex_positions = np.random.exponential(scale=mean_decay_length, size = data_size)
decay_angles = np.random.uniform(low=0, high=2*pi, size = data_size)
kaon_angles = np.random.normal(0, 1/1000, 10000)

def velocity(d,T,):
    numerator = d
    denominator = np.sqrt((T**2) + ((d**2)/c**2))
    return numerator/denominator

kaon_velocity = velocity(537, kaon_lifetime)
pion_velocity = velocity(4188, pion_lifetime)

B = pion_velocity/c
G = 1/np.sqrt(1-(B**2))

boosted_plus_four_vectors = []
boosted_zero_four_vectors = []

for i in range(data_size):
    p_x = 0
    p_y = G * pi_plus_mass * pion_velocity * np.sin(decay_angles[i])
    p_z = G * pi_plus_mass * pion_velocity * np.cos(decay_angles[i])
    p = np.sqrt((p_x**2) + (p_y**2) + (p_z**2))
    E = np.sqrt((pi_plus_mass**2)+(p**2))
    
    pi_plus_four_vector = np.array(list([E, p_x, p_y, p_z]))
    pi_zero_four_vector = np.array(list([E, p_x, -p_y, -p_z]))
    
    v_y = kaon_velocity * np.sin(kaon_angles[i])
    v_z = kaon_velocity * np.cos(kaon_angles[i])
    v = kaon_velocity
    lorentz_matrix = np.array([
        [            G, 0,                          -(G* v_y) /c,                       -(G* v_z) / c],
        [            0, 1,                                     0,                                   0],
        [-(G* v_y) / c, 0, 1 + ((G-1) * ((v_y ** 2) / (v ** 2))),    (G-1) * ((v_y * v_z) / (v ** 2))],
        [-(G* v_z) / c, 0,      (G-1) * ((v_z * v_y) / (v ** 2)), 1 + (G-1) * ((v_z ** 2) / (v ** 2))]
    ])
    
    boosted_pi_plus_four_vector = lorentz_matrix.dot(pi_plus_four_vector)
    boosted_pi_zero_four_vector = lorentz_matrix.dot(pi_zero_four_vector)
    boosted_plus_four_vectors.append(boosted_pi_plus_four_vector)
    boosted_zero_four_vectors.append(boosted_pi_zero_four_vector)

detector_positions = np.linspace(0,400, 400)
number_of_successes = []

def detections(detector_position):
    successes = 0
    for A in range(data_size):
        m = pi_plus_mass
        p_z = (boosted_plus_four_vectors[A][3])
        p_y = (boosted_plus_four_vectors[A][2])
        v_z = p_z/m
        time = abs(detector_position -decay_vertex_positions[A])/v_z
        v_y = p_y/m
        d_y = v_y * time
        if abs(d_y) < 2:
            mm = pi_zero_mass
            pz = (boosted_zero_four_vectors[A][3])
            py = (boosted_zero_four_vectors[A][2])
            vz = pz/mm
            t = abs(detector_position -decay_vertex_positions[A])/vz
            vy = py/mm
            dy = vy * t
            if abs(dy) < 2:
                successes = successes + 1
    return successes

for detector_position in detector_positions:
    number_of_successes.append(detections(detector_position))

plt.plot(detector_positions, number_of_successes, 'o', markersize=1, color='blue')
plt.title('Particle Detections as a function of Detector Position')
plt.xlabel('Detector Position in meters')
plt.ylabel('Detections')
plt.savefig('divergent kaons')
plt.show()