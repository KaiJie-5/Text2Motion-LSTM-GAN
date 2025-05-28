import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np

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

# Input: (8, 3, time_seq)
def draw_mot(plot_pose, epochs, Action, save_path='output_animation.gif'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Function to update each frame of the animation
    def update(i):
        ax.clear()
        
        # Plotting the lines and points for the current frame
        ax.plot(plot_pose[0:2, 0, i], plot_pose[0:2, 1, i], plot_pose[0:2, 2, i], c='black')
        ax.plot(plot_pose[1:3, 0, i], plot_pose[1:3, 1, i], plot_pose[1:3, 2, i], c='black')
        ax.plot(plot_pose[2:4, 0, i], plot_pose[2:4, 1, i], plot_pose[2:4, 2, i], c='black')
        ax.plot(plot_pose[3:5, 0, i], plot_pose[3:5, 1, i], plot_pose[3:5, 2, i], c='black')
        ax.plot(plot_pose[1:6:4, 0, i], plot_pose[1:6:4, 1, i], plot_pose[1:6:4, 2, i], c='black')
        ax.plot(plot_pose[5:7, 0, i], plot_pose[5:7, 1, i], plot_pose[5:7, 2, i], c='black')
        ax.plot(plot_pose[6:8, 0, i], plot_pose[6:8, 1, i], plot_pose[6:8, 2, i], c='black')

        ax.scatter(plot_pose[:, 0, i], plot_pose[:, 1, i], plot_pose[:, 2, i], c='black')

        # Setting axes properties
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-3, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(azim=-40, elev=10)
        ax.set_title(f'{i} Model_{epochs}, {Action}')

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=plot_pose.shape[2], repeat=False)

    # Save the animation as a GIF file
    writer = PillowWriter(fps=10, metadata={'title': '3D Motion Animation'})
    ani.save(save_path, writer=writer)

    plt.close(fig)
