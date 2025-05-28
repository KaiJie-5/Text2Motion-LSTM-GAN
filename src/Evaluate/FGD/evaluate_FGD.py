import numpy as np
import torch
import matplotlib.pyplot as plt

from embedding_space_evaluator import EmbeddingSpaceEvaluator
from train_AE import make_tensor

n_frames = 32
epochs = np.linspace(0,150,4, dtype= int)
case_number = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# AE model
ae_path = '../src/Evaluate/FGD/model_checkpoint_upperbody_32.bin'
fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, n_frames, device)

def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    fgd_evaluator.push_real_samples(test_data)
    fgd_evaluator.push_generated_samples(gt_data)
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    return fgd_on_feat, fdg_on_raw
'''

fgd_on_feat_array = []
fgd_on_raw_array = []

for epoch in epochs:

    # Convert to PyTorch tensors and move to the correct device
    generated_output = make_tensor(f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Generated_Data/generated_output_epoch_{epoch}.npy').to(device)
    test_action_batch = make_tensor(f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Test_Data/test_action_batch_epoch_{epoch}.npy').to(device)

    # Compute FGD
    fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, generated_output, test_action_batch)

    if fgd_on_raw >= 20000:
        fgd_on_raw = 20000

    # Print or save the results
    print(f'Epoch {epoch}')
    print(f"FGD on Feature Space: {fgd_on_feat}")
    print(f"FGD on Raw Data Space: {fgd_on_raw}")

    fgd_on_feat_array.append(fgd_on_feat)
    fgd_on_raw_array.append(fgd_on_raw)

# Visualize losses
fig, axes = plt.subplots(1,2)
axes[0].plot(epochs, fgd_on_feat_array, label='fgd_on_feat')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('FGD on Features')
axes[0].set_title('LSTM-GAN Losses')
axes[1].plot(epochs, fgd_on_raw_array, label='fgd_on_raw')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('FGD on Raw data')
axes[1].set_title('LSTM-GAN Losses')
plt.legend()
plt.savefig(f"../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/lstm_gan_FGD_losses.png")
plt.show()
'''

i = 50

# Convert to PyTorch tensors and move to the correct device
fake_motion = make_tensor(f'../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_20/LSTM_Test_Motion_Model_Epoch_150.npy').to(device)
real_motion = make_tensor(f'../Data/test_action.npy').to(device)

fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, fake_motion, real_motion)

print(f"FGD on Feature Space: {fgd_on_feat}")
print(f"FGD on Raw Data Space: {fgd_on_raw}")



