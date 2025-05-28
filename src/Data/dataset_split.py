import numpy as np


# Load and preprocess data
metadata = np.load('../Data/metadata.npz')
action = metadata['pose_array']
Embed_Script = metadata['embed_text']
Script = metadata['text']

print(Script.shape) # (31919,0)
print(Script[0])

with open("total_script.txt", "w") as f:
    for line in Script:
        f.write(str(line) + "\n")

'''
print(action.shape)
print(Embed_Script.shape)
print(Script.shape)

print("Min value:", action.min())
print("Max value:", action.max())
print("Mean:", action.mean())
print("Standard Deviation:", action.std())

print("Min value:", Embed_Script.min())
print("Max value:", Embed_Script.max())
print("Mean:", Embed_Script.mean())
print("Standard Deviation:", Embed_Script.std())

np.random.seed(42)

# Create shuffle indices to ensure alignment
shuffle_indices = np.random.permutation(Embed_Script.shape[0])

# Shuffle both arrays using the same indices
Embed_script_shuffle = Embed_Script[shuffle_indices]
action_transpose_shuffle = action[shuffle_indices]
Script_shuffle = Script[shuffle_indices]

# Define dataset sizes
total_data_size = action_transpose_shuffle.shape[0]
train_size = int(0.8 * total_data_size)
val_size = int(0.05 * total_data_size)
test_size = total_data_size - train_size - val_size

print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")

# Split the dataset
train_action = action_transpose_shuffle[:train_size]
val_action = action_transpose_shuffle[train_size:train_size + val_size]
test_action = action_transpose_shuffle[train_size + val_size:]

train_script = Embed_script_shuffle[:train_size]
val_script = Embed_script_shuffle[train_size:train_size + val_size]
test_script = Embed_script_shuffle[train_size + val_size:]

text_script = Script_shuffle[train_size + val_size:]

# Print shapes
print(train_action.shape, val_action.shape, test_action.shape)
print(train_script.shape, val_script.shape, test_script.shape)
print(text_script.shape)

# Save the data
save_path = '../Data/'
np.save(save_path + 'train_action.npy', train_action)
np.save(save_path + 'val_action.npy', val_action)
np.save(save_path + 'test_action.npy', test_action)
np.save(save_path + 'train_script.npy', train_script)
np.save(save_path + 'val_script.npy', val_script)
np.save(save_path + 'test_script.npy', test_script)
np.save(save_path + 'text_script.npy', text_script)
'''
'''
# Load datasets
train_action = np.load('../Data/train_action.npy')
val_action = np.load('../Data/val_action.npy')
test_action = np.load('../Data/test_action.npy')
train_script = np.load('../Data/train_script.npy')
val_script = np.load('../Data/val_script.npy')
test_script = np.load('../Data/test_script.npy')

# Print dataset shapes
print(f'Train Action shape: {train_action.shape}')
print(f'Validation Action shape: {val_action.shape}')
print(f'Test Action shape: {test_action.shape}')
print(f'Train Script shape: {train_script.shape}')
print(f'Validation Script shape: {val_script.shape}')
print(f'Test Script shape: {test_script.shape}')

# Compute min and max for action data
action_min = train_action.min()
action_max = train_action.max()

print(f"Train Action min: {action_min}")
print(f"Train Action min: {action_max}")

# Normalize action data
def normalize_data(data, data_min, data_max):
    return 2 * ((data - data_min) / (data_max - data_min)) - 1

train_action = normalize_data(train_action, action_min, action_max)
val_action = normalize_data(val_action, action_min, action_max)
test_action = normalize_data(test_action, action_min, action_max)

print('After normalization - Action')
print(f'Train Action Min: {train_action.min()}')
print(f'Train Action Max: {train_action.max()}')
print(f'Validation Action Min: {val_action.min()}')
print(f'Validation Action Max: {val_action.max()}')
print(f'Test Action Min: {test_action.min()}')
print(f'Test Action Max: {test_action.max()}')

# Save normalized action data
save_path = '../Data/'
np.save(save_path + 'train_action.npy', np.float32(train_action))
np.save(save_path + 'val_action.npy', np.float32(val_action))
np.save(save_path + 'test_action.npy', np.float32(test_action))

# Compute min and max for script data
script_min = train_script.min()
script_max = train_script.max()

print(f"Script Action min: {script_min}")
print(f"Script Action min: {script_max}")

# Normalize script data
train_script = normalize_data(train_script, script_min, script_max)
val_script = normalize_data(val_script, script_min, script_max)
test_script = normalize_data(test_script, script_min, script_max)

print('After normalization - Script')
print(f'Train Script Min: {train_script.min()}')
print(f'Train Script Max: {train_script.max()}')
print(f'Validation Script Min: {val_script.min()}')
print(f'Validation Script Max: {val_script.max()}')
print(f'Test Script Min: {test_script.min()}')
print(f'Test Script Max: {test_script.max()}')

# Save normalized script data
np.save(save_path + 'train_script.npy', train_script)
np.save(save_path + 'val_script.npy', val_script)
np.save(save_path + 'test_script.npy', test_script)

## Check datatype
# Load datasets
train_action = np.load('../Data/train_action.npy')
val_action = np.load('../Data/val_action.npy')
test_action = np.load('../Data/test_action.npy')
train_script = np.load('../Data/train_script.npy')
val_script = np.load('../Data/val_script.npy')
test_script = np.load('../Data/test_script.npy')

print(train_action.dtype)
print(val_action.dtype)
print(test_action.dtype)
print(train_script.dtype)
print(val_script.dtype)
print(test_script.dtype)
'''