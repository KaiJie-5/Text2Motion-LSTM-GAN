import numpy as np
import scipy.io as scio

data = np.load('../Data/metadata.npz')

#init_pose = scio.loadmat('/lyceum/kjl1a21/Filestone/Individual_Project/CoSpeech-T2A_Supervisor/data/mean_pose.mat')['mean_vector']

# Exploring the dataset

for keys in data.files:
    print(f"{keys}'s shape and Data type: " , data[keys].shape, data[keys].dtype)


############### Data's Size and Data Type #################

'''
arr_0's Shape and Data type:  (31919, 24, 32) float64
arr_1's Shape and Data type:  (31919, 300, 27) float64
arr_2's Shape and Data type:  (31919,) float64
'''

train_action = data['arr_0']
train_script = data['arr_1']
train_length = data['arr_2']

np.save('train_action', train_action)
np.save('train_script', train_script)



