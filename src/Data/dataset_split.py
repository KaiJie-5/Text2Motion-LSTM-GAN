import numpy as np

# Load and preprocess data
metadata = np.load('../Data/metadata.npz')
action = metadata['pose_array']
Embed_Script = metadata['embed_text']
Script = metadata['text']

print(Script.shape)
print(Script[0])

with open("total_script.txt", "w") as f:
    for line in Script:
        f.write(str(line) + "\n")
