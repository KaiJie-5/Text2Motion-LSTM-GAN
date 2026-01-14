import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Layer, LSTM, Dense, RepeatVector, Dropout,
                                      Concatenate, Reshape, LayerNormalization, ReLU,
                                      GlobalAveragePooling1D, Conv1D, SpectralNormalization)
from typing import Tuple
from calc_jerk_or_acceleration import tf_compute_velocity


class StackLayer(Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


class InitialStateLayer(Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, self.units))


def create_generator(
    latent_dim: int,
    text_dim: int,
    action_time_step: int,
    init_pose: np.ndarray
) -> tf.keras.Model:
    """
    Create LSTM-based Generator for gesture synthesis.

    Args:
        latent_dim: Dimension of random noise vector
        text_dim: Dimension of text embeddings (512 for Universal Sentence Encoder)
        action_time_step: Number of frames to generate (32)
        init_pose: Initial pose to condition generation

    Returns:
        Keras Model for the Generator
    """
    input_noise = Input(shape=(latent_dim,), name='noise')
    input_text = Input(shape=(text_dim,), name='text')
    init_pose_input = Input(shape=(1, 24), name='pose')

    init_pose_reshaped = Reshape((24,))(init_pose_input)

    noise_32 = RepeatVector(action_time_step)(input_noise)
    text_32 = RepeatVector(action_time_step)(input_text)
    init_pose_32 = RepeatVector(action_time_step)(init_pose_reshaped)

    combined = Concatenate()([noise_32, text_32, init_pose_32])

    x = LSTM(512, return_sequences=True)(combined)
    x = LSTM(512, return_sequences=True)(x)

    outputs = Dense(24, activation='tanh')(x)

    model = Model([input_noise, input_text, init_pose_input], outputs, name='Generator')
    return model


def create_discriminator(
    generated_input_shape: Tuple[int, int],
    text_dim: int
) -> tf.keras.Model:
    """
    Create CNN-based Discriminator for motion classification.

    Args:
        generated_input_shape: Shape of motion input (time_steps, motion_dim)
        text_dim: Dimension of text embeddings (512)

    Returns:
        Keras Model for the Discriminator
    """
    motion_input = Input(shape=generated_input_shape, name='generated_action')
    text_input = Input(shape=(text_dim,), name='text')

    motion_features = SpectralNormalization(
        Conv1D(32, kernel_size=5, strides=1, padding="same", activation=None)
    )(motion_input)
    motion_features = LayerNormalization()(motion_features)
    motion_features = ReLU()(motion_features)

    motion_features = SpectralNormalization(
        Conv1D(64, kernel_size=5, strides=2, padding="same", activation=None)
    )(motion_features)
    motion_features = LayerNormalization()(motion_features)
    motion_features = ReLU()(motion_features)

    motion_features = GlobalAveragePooling1D()(motion_features)

    combined_features = Concatenate()([motion_features, text_input])

    combined_features = Dense(128)(combined_features)
    combined_features = LayerNormalization()(combined_features)
    combined_features = ReLU()(combined_features)

    combined_features = Dropout(0.2)(combined_features)

    outputs = Dense(1, activation='sigmoid')(combined_features)

    model = Model([motion_input, text_input], outputs, name='Discriminator')

    return model


def train_d(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    noise: tf.Tensor,
    text_embeddings_batch: tf.Tensor,
    real_action_batch: tf.Tensor,
    cross_entropy: tf.keras.losses.Loss,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    batched_input: tf.Tensor
) -> Tuple[float, float, float]:
    """
    Train the Discriminator for one step.

    Args:
        generator: Generator model
        discriminator: Discriminator model
        noise: Random noise input
        text_embeddings_batch: Batch of text embeddings
        real_action_batch: Batch of real motion sequences
        cross_entropy: Loss function
        discriminator_optimizer: Optimizer for discriminator
        batched_input: Initial pose batch

    Returns:
        Tuple of (discriminator_loss, real_loss, fake_loss)
    """
    with tf.GradientTape() as disc_tape:
        fake_action_batch = generator([noise, text_embeddings_batch, batched_input], training=False)

        real_output = discriminator([real_action_batch, text_embeddings_batch], training=True)
        fake_output = discriminator([fake_action_batch, text_embeddings_batch], training=True)

        real_labels = tf.random.uniform(shape=tf.shape(real_output), minval=0.9, maxval=1.0)
        fake_labels = tf.random.uniform(shape=tf.shape(fake_output), minval=0.0, maxval=0.1)

        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)
        d_loss = real_loss + fake_loss

    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return d_loss.numpy(), real_loss.numpy(), fake_loss.numpy()


def train_g(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    noise: tf.Tensor,
    text_embeddings_batch: tf.Tensor,
    real_action_batch: tf.Tensor,
    cross_entropy: tf.keras.losses.Loss,
    mean_squared_error: tf.keras.losses.Loss,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    alpha: float,
    beta: float,
    gamma: float,
    batched_input: tf.Tensor
) -> Tuple[float, float, float]:
    """
    Train the Generator for one step.

    Args:
        generator: Generator model
        discriminator: Discriminator model
        noise: Random noise input
        text_embeddings_batch: Batch of text embeddings
        real_action_batch: Batch of real motion sequences
        cross_entropy: Binary cross entropy loss function
        mean_squared_error: MSE loss function
        generator_optimizer: Optimizer for generator
        alpha: Weight for adversarial loss
        beta: Weight for distance loss
        gamma: Weight for velocity loss
        batched_input: Initial pose batch

    Returns:
        Tuple of (adversarial_loss, distance_loss, velocity_loss)
    """
    with tf.GradientTape() as gen_tape:
        fake_action_batch = generator([noise, text_embeddings_batch, batched_input], training=True)

        fake_output = discriminator([fake_action_batch, text_embeddings_batch], training=False)

        real_labels = tf.ones_like(fake_output)

        real_vel = tf_compute_velocity(real_action_batch)
        fake_vel = tf_compute_velocity(fake_action_batch)
        vel_loss = tf.reduce_mean(tf.abs(fake_vel - real_vel))

        g_loss = cross_entropy(real_labels, fake_output)
        g_distance_loss = mean_squared_error(real_action_batch, fake_action_batch)

        total_g_loss = alpha * g_loss + beta * g_distance_loss + gamma * vel_loss

    gradients_of_generator = gen_tape.gradient(total_g_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return g_loss.numpy(), g_distance_loss.numpy(), gamma * vel_loss.numpy()
