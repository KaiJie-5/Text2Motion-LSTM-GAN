"""
Plots the experimental results after calculating motion statistics
Expects that calc_distance was run before this script

@author: Taras Kucherenko
"""

import matplotlib.pyplot as plt
import csv
import numpy as np

def read_joint_names(filename):
    with open(filename, 'r') as f:
        org = f.read()
        joint_names = org.split(',')

    return joint_names

def read_csv(filename):

    x=[]
    y=[]
    total_sum = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            x.append(float(row[0]))
            next_val = float(row[-2]) + float(row[-5])
            y.append(next_val)
            total_sum += next_val

    return np.array(x), np.array(y) / total_sum


plt.rcParams.update({'font.size': 20})

plt.rcParams.update({'pdf.fonttype': 42, 'font.family':'sans'})

type = "vel"
case_number = 5

original_filename = f'../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_{case_number}/original/hmd_'+type+'_1.csv'

x,original = read_csv(original_filename)

# Get Full model results
feature_filename = f'../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_5/predicted/hmd_' + type + '_1.csv'
_, full_model = read_csv(feature_filename)

# Get No Vel Loss model results
_, no_vel_loss = read_csv('../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_15/predicted/hmd_' + type + '_1.csv')

# Get Only Adversarial Loss model results
_, only_adv_loss = read_csv('../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_14/predicted/hmd_' + type + '_1.csv')

# Get distance and adv loss model results
_, vel_distance_loss = read_csv('../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_16/predicted/hmd_' + type + '_1.csv')

# Plot the results
plt.plot(x,original,linewidth=7.0, label='Real Motion', color='Blue')
plt.plot(x,full_model , linewidth=7.0, label='Adv + Dist Loss + Vel Loss', color='C6')
plt.plot(x,no_vel_loss , linewidth=7.0, label='Adv + Dist Loss', color='C1')
plt.plot(x,only_adv_loss , linewidth=7.0, label='Adv Loss Only', color='C2')
plt.plot(x,vel_distance_loss , linewidth=7.0, label='Dist + Vel Loss', color='C3')

plt.xlabel("Velocity ($cm$/$s$)", size=20)
plt.ylabel('Frequency (%)', size=20)

plt.xticks(size=20)

plt.xticks(np.arange(0, 1.0 + 0.1, 0.1), size=15)  # Ticks every 0.1 cm/s
plt.xlim(0, 1.0)

leg = plt.legend(prop={'size': 20})

figure = plt.gcf() # get current figure
figure.set_size_inches(23, 13)
plt.savefig(f"../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_{case_number}/myplot.pdf") #, dpi = 100)