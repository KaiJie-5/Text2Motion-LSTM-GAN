import numpy as np
import tensorflow as tf
from calc_jerk_or_acceleration import compute_jerks, compute_acceleration, compute_velocity
from calc_errors import APE

MAE = tf.keras.losses.MeanAbsoluteError()

case_number = 10
i = 150

'''
#For validation set
fake_motion = np.load(f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Generated_Data/generated_output_epoch_{i}.npy') 
real_motion = np.load(f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Test_Data/test_action_batch_epoch_{i}.npy')
'''

# For test set
fake_test_motion = np.load('../Baseline_Model/CoSpeech/exec/CoSpeech_Motion_List_Epoch_400.npy')
test_motion = np.load('../Data/test_action.npy')

# Compute average jerk
real_jerks = compute_jerks(test_motion)
fake_jerks = compute_jerks(fake_test_motion)

# Compute average acceleration
real_acc = compute_acceleration(test_motion)
fake_acc = compute_acceleration(fake_test_motion)

# Compute average acceleration
real_vel = compute_velocity(test_motion)
fake_vel = compute_velocity(fake_test_motion)

'''
print("****Real Motion****\n")
print(f"Mean of Jerks: {np.mean(real_jerks)}\n")
print(f"Standard Deviation of Jerks: {np.std(real_jerks)}\n")
print(f"Mean Acceleration: {np.mean(real_acc)}\n")
print(f"Standard Deviation of Acceleration: {np.std(real_acc)}\n")
print(f"Mean Velocity: {np.mean(real_vel)}\n")
print(f"Standard Deviation of Velocity: {np.std(real_vel)}\n")
'''

print("****Fake motion****\n")
print(f"Mean of Jerks: {np.mean(fake_jerks)}\n")
print(f"Standard Deviation of Jerks: {np.std(fake_jerks)}\n")
print(f"Mean Acceleration: {np.mean(fake_acc)}\n")
print(f"Standard Deviation of Acceleration: {np.std(fake_acc)}\n")
print(f"Mean Velocity: {np.mean(fake_vel)}\n")
print(f"Standard Deviation of Velocity: {np.std(fake_vel)}\n")

# Calculate the MAE and APE
mae = MAE(test_motion,fake_test_motion)
ape = APE(test_motion,fake_test_motion)

print(f"MAE: {mae}\n")
print(f"Mean APE: {np.mean(ape)}\n")
print(f"Standard Deviation APE: {np.std(ape)}\n")