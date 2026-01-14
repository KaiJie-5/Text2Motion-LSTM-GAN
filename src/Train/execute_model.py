import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as scio
from pathlib import Path
from structure_GAN import create_Generator, create_discriminator, train_d, train_g

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load your data
train_script = np.load(config.DATA_DIR / 'train_script.npy')
train_action = np.load(config.DATA_DIR / 'train_action.npy')
val_action = np.load(config.DATA_DIR / 'val_action.npy')
val_script = np.load(config.DATA_DIR / 'val_script.npy')
init_pose = scio.loadmat(config.DATA_DIR / 'mean_pose.mat')['mean_vector']
init_pose = np.transpose(init_pose,(1,0))

print(f'Train Action Min :{train_action.min()}')
print(f'Train Action Max :{train_action.max()}')
print(f'Val Action Min :{val_action.min()}')
print(f'Val Action Max :{val_action.max()}')
print(f'Train Script Min :{train_script.min()}')
print(f'Train Script Max :{train_script.max()}')
print(f'Val Script Min :{val_script.min()}')
print(f'Val Script Max :{val_script.max()}')

print(train_script.shape)
print(train_action.shape)
print(val_action.shape)
print(val_script.shape)
print(init_pose.shape)

# Hyperparameters from config
latent_dim = config.LATENT_DIM
text_dim = config.TEXT_DIM
action_time_steps = config.ACTION_TIME_STEPS
dim_action = config.DIM_ACTION
epochs = config.EPOCHS
batch_size = config.BATCH_SIZE
alpha = config.ALPHA
beta = config.BETA
gamma = config.GAMMA
case_number = 5

# Create a TensorFlow Dataset from training data
dataset = tf.data.Dataset.from_tensor_slices((train_script, train_action))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Create a TensorFlow Dataset from validation data
val_dataset = tf.data.Dataset.from_tensor_slices((val_script, val_action))
val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

batched_input = tf.tile(tf.expand_dims(init_pose, axis=0), [batch_size, 1, 1])
print(batched_input.shape)

model_saved_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/'
generated_output_saved_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Generated_Data/'
test_action_saved_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Test_Data/'

# List of paths to check and create
paths = [
    model_saved_path,
    generated_output_saved_path,
    test_action_saved_path
]

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)

# Create models
generator = create_Generator(latent_dim, text_dim, action_time_steps,init_pose)
discriminator = create_discriminator((action_time_steps, dim_action), text_dim)

print(generator.summary())
print(discriminator.summary())

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_G)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_D)

# Define loss function  
cross_entropy = tf.keras.losses.BinaryCrossentropy()
mean_squared_error = tf.keras.losses.MeanSquaredError()

epoch_d_loss = []
epoch_g_loss = []
epoch_g_distance_loss = []
epoch_acceleration_loss = []
epoch_real_loss = []
epoch_fake_loss = []

d_loss_array = []
g_loss_array = []
real_loss_array = []
fake_loss_array = []
g_distance_loss_array = []
validation_loss_array = []
acceleration_loss_array = []

# Training loop
for epoch in range(epochs + 1):

    all_val_generated = []
    all_val_real = []

    for train_text_batch, train_action_batch in dataset:

        noise = tf.random.normal([batch_size, latent_dim])

        discriminator_loss, r_loss, f_loss = train_d(generator, discriminator, noise, train_text_batch, train_action_batch, cross_entropy, discriminator_optimizer,batched_input)
        generator_loss, generator_distance_loss, acceleration_loss = train_g(generator, discriminator, noise, train_text_batch, train_action_batch, cross_entropy, mean_squared_error, generator_optimizer, alpha, beta, gamma, batched_input)

        epoch_d_loss.append(discriminator_loss)
        epoch_g_loss.append(generator_loss)
        epoch_g_distance_loss.append(generator_distance_loss)
        epoch_acceleration_loss.append(acceleration_loss)
        epoch_real_loss.append(r_loss)
        epoch_fake_loss.append(f_loss)
    
    d_loss_array.append(np.mean(epoch_d_loss))
    g_loss_array.append(np.mean(epoch_g_loss))
    g_distance_loss_array.append(np.mean(epoch_g_distance_loss))
    acceleration_loss_array.append(np.mean(epoch_acceleration_loss))
    real_loss_array.append(np.mean(epoch_real_loss))
    fake_loss_array.append(np.mean(epoch_fake_loss))

    for val_text_batch, val_action_batch in val_dataset:

        current_batch_size = val_text_batch.shape[0]
        noise = tf.random.normal([current_batch_size, latent_dim])

        generated_output = generator([noise,val_text_batch,batched_input], training=False)

        #Save the test and generated action batch for evaluation

        all_val_generated.append(generated_output.numpy())
        all_val_real.append(val_action_batch.numpy())

    # Save the model periodically
    if epoch % config.CHECKPOINT_FREQUENCY == 0:
        generator.save(f"../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/model_epoch_{epoch}.keras")

        # Save to files
        np.save(f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Generated_Data/generated_output_epoch_{epoch}.npy', np.concatenate(all_val_generated))
        np.save(f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Test_Data/test_action_batch_epoch_{epoch}.npy', np.concatenate(all_val_real))

    print(f"{epoch}/{epochs} [Real loss: {r_loss}] [Fake loss: {f_loss}] [G loss: {generator_loss}] [G_distance_loss: {generator_distance_loss}] [Velocity Loss: {acceleration_loss}]")

# Visualize losses
plt.figure(figsize=(12, 6))
plt.plot(range(epochs + 1), d_loss_array, label='Discriminator Loss')
plt.plot(range(epochs + 1), g_loss_array, label='Generator Cross Entropy Loss')
plt.plot(range(epochs + 1), g_distance_loss_array, label='Generator Distance Loss')
plt.plot(range(epochs + 1), real_loss_array, label='Real Loss')
plt.plot(range(epochs + 1), fake_loss_array, label='Fake Loss')
plt.plot(range(epochs + 1), acceleration_loss_array, label='Velocity Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('LSTM-GAN Training Losses')
plt.legend()
plt.savefig(f"../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/lstm_gan_training_losses.png")
#plt.show()
