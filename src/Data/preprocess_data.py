import tensorflow_hub as hub
import os
import scipy.io as scio
import numpy as np
import kagglehub

'''
# Download latest version
path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder")

print(path)
'''

model_path = '../2'

#Load the model

def load_encoder(model_path):
    #Output's dimension = 512
    encoder = hub.load(model_path)
    return encoder

encoder = load_encoder(model_path)

'''
def preprocess():
    # First, read the overall text file to compute maximum sentence length (optional)
    with open('../Data/total_script.txt', 'r') as f:
        lines = f.readlines()
    # Compute the maximum number of words in any line (this may be useful for later processing)
    max_length = max(len(line.strip().split()) for line in lines)
    print('The maximum sentence length is %d' % max_length)

    # Define paths for the pose and script data
    pose_path = '../Data/pose'
    script_path = '../Data/script'
    pose_files = [f for f in os.listdir(pose_path) if os.path.isfile(os.path.join(pose_path, f))]
    
    total_pose_list = []
    total_text_list = []
    total_embed_text_list = []

    # Loop through each pose file and its corresponding script file.
    for idx, p_file in enumerate(pose_files):
        print('processing %dth data...' % idx)
        # Load the pose data from the .mat file
        curr_pred = scio.loadmat(os.path.join(pose_path, p_file))['pred_vector']
        script_filename = 'script_' + p_file[5:9] + '.txt'
        curr_script_path = os.path.join(script_path, script_filename)
        
        # Open the corresponding script file and read its lines
        with open(curr_script_path, 'r') as s_f:
            script_lines = s_f.readlines()
            for line in script_lines:
                text_line = line.strip().lower()
                embed_text = encoder([text_line]).numpy()[0]
                total_embed_text_list.append(embed_text)
                total_text_list.append(text_line)
                total_pose_list.append(curr_pred.T)

    # Convert the list of pose data to a NumPy array.
    pose_array = np.asarray(total_pose_list)
    print("Pose data shape:", pose_array.shape)

    print('DATA READY')

    # Save the pose data, raw text lines, sentence lengths, and maximum sentence length to an npz file.
    np.savez('../Data/metadata.npz', pose_array=pose_array, text=total_text_list,
             embed_text = total_embed_text_list)
    print('DATA SAVED')

if __name__ == "__main__":
    preprocess()
'''

