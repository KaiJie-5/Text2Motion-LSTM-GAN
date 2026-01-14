import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as scio
import argparse
import logging
from pathlib import Path
from typing import Tuple
from structure_GAN import create_generator, create_discriminator, train_d, train_g

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and validation data.

    Returns:
        Tuple of train_script, train_action, val_script, val_action, init_pose

    Raises:
        FileNotFoundError: If data files are not found
        Exception: If data loading fails
    """
    try:
        logger.info("Loading training and validation data")

        train_script_path = config.DATA_DIR / 'train_script.npy'
        train_action_path = config.DATA_DIR / 'train_action.npy'
        val_action_path = config.DATA_DIR / 'val_action.npy'
        val_script_path = config.DATA_DIR / 'val_script.npy'
        mean_pose_path = config.DATA_DIR / 'mean_pose.mat'

        if not train_script_path.exists():
            raise FileNotFoundError(f"Train script not found: {train_script_path}")
        if not train_action_path.exists():
            raise FileNotFoundError(f"Train action not found: {train_action_path}")
        if not val_action_path.exists():
            raise FileNotFoundError(f"Validation action not found: {val_action_path}")
        if not val_script_path.exists():
            raise FileNotFoundError(f"Validation script not found: {val_script_path}")
        if not mean_pose_path.exists():
            raise FileNotFoundError(f"Mean pose not found: {mean_pose_path}")

        train_script = np.load(train_script_path)
        train_action = np.load(train_action_path)
        val_action = np.load(val_action_path)
        val_script = np.load(val_script_path)
        init_pose = scio.loadmat(mean_pose_path)['mean_vector']
        init_pose = np.transpose(init_pose, (1, 0))

        logger.info(f"Train script shape: {train_script.shape}")
        logger.info(f"Train action shape: {train_action.shape}")
        logger.info(f"Validation script shape: {val_script.shape}")
        logger.info(f"Validation action shape: {val_action.shape}")
        logger.info(f"Initial pose shape: {init_pose.shape}")

        logger.info(f"Train Action - Min: {train_action.min():.4f}, Max: {train_action.max():.4f}")
        logger.info(f"Val Action - Min: {val_action.min():.4f}, Max: {val_action.max():.4f}")
        logger.info(f"Train Script - Min: {train_script.min():.4f}, Max: {train_script.max():.4f}")
        logger.info(f"Val Script - Min: {val_script.min():.4f}, Max: {val_script.max():.4f}")

        return train_script, train_action, val_script, val_action, init_pose

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def setup_output_directories(case_number: int) -> Tuple[str, str, str]:
    """
    Create output directories for models and results.

    Args:
        case_number: Case number for the experiment

    Returns:
        Tuple of (model_path, generated_output_path, test_action_path)
    """
    try:
        model_saved_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/'
        generated_output_saved_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Generated_Data/'
        test_action_saved_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Test_Data/'

        paths = [model_saved_path, generated_output_saved_path, test_action_saved_path]

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")

        return model_saved_path, generated_output_saved_path, test_action_saved_path

    except Exception as e:
        logger.error(f"Failed to create output directories: {e}")
        raise


def train_model(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    cross_entropy: tf.keras.losses.Loss,
    mean_squared_error: tf.keras.losses.Loss,
    batched_input: tf.Tensor,
    case_number: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    alpha: float,
    beta: float,
    gamma: float
) -> None:
    """
    Train the LSTM-GAN model.

    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataset: Training dataset
        val_dataset: Validation dataset
        generator_optimizer: Optimizer for generator
        discriminator_optimizer: Optimizer for discriminator
        cross_entropy: Binary cross entropy loss
        mean_squared_error: MSE loss
        batched_input: Initial pose batch
        case_number: Experiment case number
        epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Latent dimension for noise
        alpha: Adversarial loss weight
        beta: Distance loss weight
        gamma: Velocity loss weight
    """
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
    acceleration_loss_array = []

    logger.info("Starting training")

    for epoch in range(epochs + 1):
        all_val_generated = []
        all_val_real = []

        for train_text_batch, train_action_batch in dataset:
            noise = tf.random.normal([batch_size, latent_dim])

            discriminator_loss, r_loss, f_loss = train_d(
                generator, discriminator, noise, train_text_batch,
                train_action_batch, cross_entropy, discriminator_optimizer, batched_input
            )
            generator_loss, generator_distance_loss, acceleration_loss = train_g(
                generator, discriminator, noise, train_text_batch, train_action_batch,
                cross_entropy, mean_squared_error, generator_optimizer,
                alpha, beta, gamma, batched_input
            )

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

            generated_output = generator([noise, val_text_batch, batched_input], training=False)

            all_val_generated.append(generated_output.numpy())
            all_val_real.append(val_action_batch.numpy())

        if epoch % config.CHECKPOINT_FREQUENCY == 0:
            try:
                model_path = f"../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/model_epoch_{epoch}.keras"
                generator.save(model_path)
                logger.info(f"Saved model at epoch {epoch}")

                generated_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Generated_Data/generated_output_epoch_{epoch}.npy'
                test_path = f'../Hyperparameter_Tuning_and_Ablation_Study/Test_and_Generated_Data/Case_{case_number}/Test_Data/test_action_batch_epoch_{epoch}.npy'

                np.save(generated_path, np.concatenate(all_val_generated))
                np.save(test_path, np.concatenate(all_val_real))
                logger.info(f"Saved generated outputs at epoch {epoch}")

            except Exception as e:
                logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Real loss: {r_loss:.4f}, Fake loss: {f_loss:.4f}, "
            f"G loss: {generator_loss:.4f}, G distance loss: {generator_distance_loss:.4f}, "
            f"Velocity loss: {acceleration_loss:.4f}"
        )

    try:
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

        plot_path = f"../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/lstm_gan_training_losses.png"
        plt.savefig(plot_path)
        logger.info(f"Saved training loss plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save loss plot: {e}")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train LSTM-GAN for gesture generation')
    parser.add_argument(
        '--case',
        type=int,
        default=config.DEFAULT_CASE_NUMBER,
        help=f'Case number for experiment (default: {config.DEFAULT_CASE_NUMBER})'
    )
    args = parser.parse_args()

    try:
        logger.info(f"Starting LSTM-GAN training for Case {args.case}")
        logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

        train_script, train_action, val_script, val_action, init_pose = load_data()

        latent_dim = config.LATENT_DIM
        text_dim = config.TEXT_DIM
        action_time_steps = config.ACTION_TIME_STEPS
        dim_action = config.DIM_ACTION
        epochs = config.EPOCHS
        batch_size = config.BATCH_SIZE
        alpha = config.ALPHA
        beta = config.BETA
        gamma = config.GAMMA
        case_number = args.case

        dataset = tf.data.Dataset.from_tensor_slices((train_script, train_action))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_script, val_action))
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        batched_input = tf.tile(tf.expand_dims(init_pose, axis=0), [batch_size, 1, 1])
        logger.info(f"Batched input shape: {batched_input.shape}")

        setup_output_directories(case_number)

        logger.info("Creating models")
        generator = create_generator(latent_dim, text_dim, action_time_steps, init_pose)
        discriminator = create_discriminator((action_time_steps, dim_action), text_dim)

        logger.info("Generator summary:")
        generator.summary(print_fn=logger.info)
        logger.info("Discriminator summary:")
        discriminator.summary(print_fn=logger.info)

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_G)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_D)

        cross_entropy = tf.keras.losses.BinaryCrossentropy()
        mean_squared_error = tf.keras.losses.MeanSquaredError()

        train_model(
            generator, discriminator, dataset, val_dataset,
            generator_optimizer, discriminator_optimizer,
            cross_entropy, mean_squared_error, batched_input,
            case_number, epochs, batch_size, latent_dim,
            alpha, beta, gamma
        )

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
