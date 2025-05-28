import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import tensorflow as tf

test_action = np.load('../Data/test_action.npy') #(6384, 32, 24)
test_script = np.load('../Data/test_script.npy') #(6384, 512)
init_pose = scio.loadmat('../Data/mean_pose.mat')['mean_vector']
init_pose = np.transpose(init_pose,(1,0))
init_input = tf.tile(tf.expand_dims(init_pose, axis=0), [1, 1, 1])
print(init_input.shape)


idx = np.random.randint(0, test_action.shape[0], 1)
case_number = 5
model_epoch = 150
latent_dim = 20
save_Real_path = f"../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/static_Real_{idx}_frame.png"
save_Fake_path = f"../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/static_Fake_{idx}_frame_1.png"

# Construct 3D Joints
# Input: (24, time_seq)
# Output: (8, 3, time_seq)
def construct_joint3D(plot_vec):
    plot_vec = np.reshape(plot_vec, [8, 3, -1])

    mean_len = [0.6, 0.7, 0.9, 0.9, 0.7, 0.9, 0.9]

    plot_pose = np.zeros(plot_vec.shape)
    plot_pose[1, :, :] = plot_vec[0, :, :];
    plot_pose[0, :, :] = plot_pose[1, :, :]+\
                         mean_len[0] * np.divide(plot_vec[1, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[1, :, :], axis=0), (3, 1)))
    plot_pose[2, :, :] = plot_pose[1, :, :]+\
                         mean_len[1] * np.divide(plot_vec[2, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[2, :, :], axis=0), (3, 1)))
    plot_pose[3, :, :] = plot_pose[2, :, :]+\
                         mean_len[2] * np.divide(plot_vec[3, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[3, :, :], axis=0), (3, 1)))
    plot_pose[4, :, :] = plot_pose[3, :, :]+\
                         mean_len[3] * np.divide(plot_vec[4, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[4, :, :], axis=0), (3, 1)))
    plot_pose[5, :, :] = plot_pose[1, :, :]+\
                         mean_len[4] * np.divide(plot_vec[5, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[5, :, :], axis=0), (3, 1)))
    plot_pose[6, :, :] = plot_pose[5, :, :]+\
                         mean_len[5] * np.divide(plot_vec[6, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[6, :, :], axis=0), (3, 1)))
    plot_pose[7, :, :] = plot_pose[6, :, :]+\
                         mean_len[6] * np.divide(plot_vec[7, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[7, :, :], axis=0), (3, 1)))
    
    # plot the virtal central_hip, left hip, right hip
    v2h = np.array([plot_pose[2, 0, :],plot_pose[2, 1, :], plot_pose[2, 2, :]-1.5])
    v5h = np.array([plot_pose[5, 0, :],plot_pose[5, 1, :], plot_pose[5, 2, :]-1.5])
    
    pelvis = (v2h+v5h)/2
    pelvis = np.expand_dims(pelvis, axis=0)
    plot_pose = np.concatenate([plot_pose, pelvis])
    return plot_pose

def draw_static_frames(plot_pose, save_path):
    num_frames = plot_pose.shape[2]
    cols = num_frames
    rows = 1

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, subplot_kw={'projection': '3d'}, figsize=(16, 4 * rows))
    axes = axes.ravel()  # Flatten axes for easy iteration

    for i in range(num_frames):
        ax = axes[i]

        # Plot lines and points for the current frame
        ax.plot(plot_pose[0:2, 0, i], plot_pose[0:2, 1, i], plot_pose[0:2, 2, i], c='black')
        ax.plot(plot_pose[1:3, 0, i], plot_pose[1:3, 1, i], plot_pose[1:3, 2, i], c='black')
        ax.plot(plot_pose[2:4, 0, i], plot_pose[2:4, 1, i], plot_pose[2:4, 2, i], c='black')
        ax.plot(plot_pose[3:5, 0, i], plot_pose[3:5, 1, i], plot_pose[3:5, 2, i], c='black')
        ax.plot(plot_pose[1:6:4, 0, i], plot_pose[1:6:4, 1, i], plot_pose[1:6:4, 2, i], c='black')
        ax.plot(plot_pose[5:7, 0, i], plot_pose[5:7, 1, i], plot_pose[5:7, 2, i], c='black')
        ax.plot(plot_pose[6:8, 0, i], plot_pose[6:8, 1, i], plot_pose[6:8, 2, i], c='black')

        ax.scatter(plot_pose[:, 0, i], plot_pose[:, 1, i], plot_pose[:, 2, i], c='black')

        # Remove grid, ticks, and labels
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.view_init(azim=0, elev=0)
        ax.set_title(f'Frame {i}')

    # Hide unused subplots
    for j in range(num_frames, len(axes)):
        axes[j].axis('off')

    # Save the grid of static frames to a PDF file
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# Example Usage
# Assuming `plot_pose` is a numpy array of shape (8, 3, time_seq)
# draw_static_frames(plot_pose, epochs=450, Action='Fake Action', save_path='static_frames.pdf')

real_motion = test_action[idx]
print(real_motion.shape)
real_motion_transpose =np.transpose(real_motion,(0,2,1))
script = test_script[idx]

generator = tf.keras.models.load_model(f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/model_epoch_{model_epoch}.keras')
script_convert = tf.convert_to_tensor(script, dtype=tf.float32)
noise = tf.random.normal([1,latent_dim])

print(noise.shape)
print(script_convert.shape)
print(init_input.shape)

generated_motion = generator.predict([noise,script_convert,init_input])

print(generated_motion.shape)
generated_motion = np.transpose(generated_motion,(0,2,1))

# Select specific timeframes: 0, 4, 8, 12, 16, 20, 24, 28, 31
selected_frames = [0, 4, 8, 12, 16, 20, 24, 28, 31]

# Filter the real_motion_transpose and generated_motion to only include selected frames
real_motion_filtered = real_motion_transpose[:, :, selected_frames]
generated_motion_filtered = generated_motion[:, :, selected_frames]

# Reconstruct the joint 3D positions for the filtered frames
real_action_filtered = construct_joint3D(real_motion_filtered)
fake_action_filtered = construct_joint3D(generated_motion_filtered)

# Draw Real and Fake motion for the selected frames
draw_static_frames(real_action_filtered, save_Real_path)
draw_static_frames(fake_action_filtered, save_Fake_path)

