'''
calculate mean and std of the natural (*NA) condition files
'''

import numpy as np

all_data = np.load('../Data/train_action.npy')

print(all_data.shape)

# Reshape to combine samples and time steps
reshaped_data = all_data.reshape(-1, all_data.shape[2])  # Shape: (31919*32, 24)
print(reshaped_data.shape)  # Shape should now be (1021408, 24)

mean = np.mean(reshaped_data, axis=0)
std = np.std(reshaped_data, axis=0)
print(mean.shape)
print(std.shape)

print(*mean, sep=',')
print(*std, sep=',')
