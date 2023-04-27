import numpy as np
import matplotlib.pyplot as plt
decay_length_data = np.loadtxt('dec_lengths.txt')
total_mean = np.mean(decay_length_data)

# plt.hist(decay_length_data, bins=1000)
# plt.show()

def kaon_mean_decay_length(u, u_pi):
    return (u -0.84*u_pi)/0.16
    
print(kaon_mean_decay_length(total_mean, 4188))
print(total_mean)
print(4188)
print('')

# to check
a = list(np.random.exponential(scale=4188, size=84000))
b = list(np.random.exponential(scale=537, size=16000))
c = a + b
print(np.mean(b))

print(np.mean(c))
print(np.mean(a))

# plt.hist(c, bins=1000)
# plt.show()
