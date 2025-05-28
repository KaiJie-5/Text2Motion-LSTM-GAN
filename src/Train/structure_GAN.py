import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Layer,LSTM, Dense, RepeatVector,Dropout,
                                      Concatenate,Reshape, LayerNormalization, ReLU,
                                      GlobalAveragePooling1D, Conv1D,SpectralNormalization)
from calk_jerk_or_acceleration import tf_compute_velocity

class StackLayer(Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)
    
class InitialStateLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]  
        return tf.zeros((batch_size, self.units))

# Define the generator model
def create_Generator(latent_dim, text_dim, action_time_step, init_pose):
    input_noise = Input(shape=(latent_dim,), name='noise')
    input_text = Input(shape=(text_dim,), name='text')
    init_pose = Input(shape=(1,24), name='pose')

    #Reshape init_pose
    init_pose_reshaped = Reshape((24,))(init_pose)

    # Repeat inputs for all timesteps
    noise_32 = RepeatVector(action_time_step)(input_noise)
    text_32 = RepeatVector(action_time_step)(input_text)
    init_pose_32 = RepeatVector(action_time_step)(init_pose_reshaped)

    combined = Concatenate()([noise_32, text_32, init_pose_32])

    # Single LSTM processing
    x = LSTM(512, return_sequences=True)(combined)
    x = LSTM(512, return_sequences=True)(x)

    outputs = Dense(24, activation='tanh')(x)

    model = Model([input_noise,input_text,init_pose], outputs, name='Generator')
    return model

def create_discriminator(generated_input_shape, text_dim):
    # Motion Input (Time, Motion_Dim)
    motion_input = Input(shape=generated_input_shape, name='generated_action')
    
    # Text Input
    text_input = Input(shape=(text_dim,), name='text')

    # Apply 1D Conv on Motion Only
    motion_features = SpectralNormalization(Conv1D(32, kernel_size=5, strides=1, padding="same", activation=None))(motion_input)
    motion_features = LayerNormalization()(motion_features)
    motion_features = ReLU()(motion_features)

    motion_features = SpectralNormalization(Conv1D(64, kernel_size=5, strides=2, padding="same", activation=None))(motion_features)
    motion_features = LayerNormalization()(motion_features)
    motion_features = ReLU()(motion_features)

    # Global Average Pooling to Reduce Temporal Dimension
    motion_features = GlobalAveragePooling1D()(motion_features)

    # Concatenate Motion and Text Features
    combined_features = Concatenate()([motion_features, text_input])

    # Fully Connected Layers
    combined_features = Dense(128)(combined_features)
    combined_features = LayerNormalization()(combined_features)
    combined_features = ReLU()(combined_features)

    combined_features = Dropout(0.2)(combined_features)

    # Output Layer (Real/Fake Classification)
    outputs = Dense(1, activation='sigmoid')(combined_features)

    # Create Model
    model = Model([motion_input, text_input], outputs, name='Discriminator')

    return model

# Train Discriminator
def train_d(generator, discriminator, noise, text_embeddings_batch, real_action_batch, cross_entropy, discriminator_optimizer,batched_input):
    with tf.GradientTape() as disc_tape:
        # Generate fake actions
        fake_action_batch = generator([noise, text_embeddings_batch, batched_input], training=False)

        # Discriminator outputs
        real_output = discriminator([real_action_batch, text_embeddings_batch], training=True)
        fake_output = discriminator([fake_action_batch, text_embeddings_batch], training=True)

        # Generate real labels with random values between 0.9 and 1.0
        real_labels = tf.random.uniform(shape=tf.shape(real_output), minval=0.9, maxval=1.0)

        # Generate fake labels with random values between 0.0 and 0.1
        fake_labels = tf.random.uniform(shape=tf.shape(fake_output), minval=0.0, maxval=0.1)

        # Discriminator loss
        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)
        d_loss = real_loss + fake_loss

    # Compute gradients
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    # Apply gradients
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return d_loss.numpy(), real_loss.numpy(), fake_loss.numpy()

def train_g(generator, discriminator, noise, text_embeddings_batch, real_action_batch, cross_entropy, mean_squared_error, generator_optimizer, alpha, beta, gamma, batched_input):
    with tf.GradientTape() as gen_tape:
        # Generate fake actions
        fake_action_batch = generator([noise, text_embeddings_batch, batched_input], training=True)

        # Discriminator outputs
        fake_output = discriminator([fake_action_batch, text_embeddings_batch], training=False)

        real_labels = tf.ones_like(fake_output)

        # Compute acceleration loss
        real_vel = tf_compute_velocity(real_action_batch)
        fake_vel = tf_compute_velocity(fake_action_batch)
        vel_loss = tf.reduce_mean(tf.abs(fake_vel - real_vel))

        # Generator loss
        g_loss = cross_entropy(real_labels, fake_output)
        g_distance_loss = mean_squared_error(real_action_batch,fake_action_batch)

        total_g_loss = alpha*g_loss + beta*g_distance_loss + gamma*vel_loss

    # Compute gradients
    gradients_of_generator = gen_tape.gradient(total_g_loss, generator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return g_loss.numpy(), g_distance_loss.numpy(), gamma*vel_loss.numpy()