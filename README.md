# Robot Gesture Generation using LSTM-GAN

This repository contains the implementation of an LSTM-based Generative Adversarial Network (GAN) for synthesizing human-like gestures from text inputs. The project is part of my final year individual research at the University of Southampton, focusing on communicative robot motion generation.

## Project Overview

The goal of this project is to generate realistic gesture sequences conditioned on textual descriptions. The model architecture consists of:
- **Text Encoder**: Universal Sentence Encoder that embeds input text into 512-dimensional vectors
- **Generator (LSTM-GAN)**: Produces 32-frame gesture sequences from noise, text embeddings, and initial pose
- **Discriminator**: CNN-based classifier that distinguishes between real and synthetic motion sequences

## Key Features

- Multi-objective loss function (adversarial, distance, and velocity losses)
- Label smoothing for stable GAN training
- Spectral normalization in discriminator
- Support for 20 different experimental configurations
- Real-world deployment on Pepper robot
- Comprehensive evaluation metrics (MAE, APE, FGD, Jerk, Acceleration)

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 12.1 or 12.2 (for GPU support)
- Git LFS (for large files)

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Text2Motion-LSTM-GAN.git
cd Text2Motion-LSTM-GAN

# Install the package and dependencies
pip install -e .
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/Text2Motion-LSTM-GAN.git
cd Text2Motion-LSTM-GAN

# Create conda environment
conda env create -f environment.yml
conda activate myenv
```

### Option 3: Manual installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Text2Motion-LSTM-GAN.git
cd Text2Motion-LSTM-GAN

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training the Model

#### Basic training with default settings:

```bash
cd src/Train
python execute_model.py
```

This will train the model with default hyperparameters (case 5, 150 epochs, batch size 32).

#### Training with custom case number:

```bash
cd src/Train
python execute_model.py --case 10
```

#### View all available options:

```bash
python execute_model.py --help
```

### Training on HPC Cluster

If you have access to an HPC cluster with SLURM:

```bash
# Edit job.slurm to set your email and preferences
sbatch job.slurm
```

## Configuration

All hyperparameters and paths are centralized in `config.py`:

```python
# Model hyperparameters
LATENT_DIM = 20              # Noise dimension
TEXT_DIM = 512               # Text embedding dimension
ACTION_TIME_STEPS = 32       # Sequence length
EPOCHS = 150                 # Training epochs
BATCH_SIZE = 32              # Batch size

# Loss weights
ALPHA = 1                    # Adversarial loss weight
BETA = 10                    # Distance loss weight
GAMMA = 5                    # Velocity loss weight

# Learning rates
LEARNING_RATE_G = 0.00002    # Generator learning rate
LEARNING_RATE_D = 0.00002    # Discriminator learning rate
```

To change settings, edit `config.py` instead of modifying code files.

## Data Format

### Input Data Structure

```
Data/
├── pose/                    # Raw pose files (.mat format)
│   └── pose_XXXX.mat       # 8 joints x 3 coordinates x 32 timesteps
├── script/                  # Text descriptions
│   └── script_XXXX.txt     # Multiple text descriptions per pose
├── train_action.npy        # Preprocessed training motions (25535, 32, 24)
├── val_action.npy          # Validation motions (1595, 32, 24)
├── test_action.npy         # Test motions (6384, 32, 24)
├── train_script.npy        # Training text embeddings (25535, 512)
├── val_script.npy          # Validation text embeddings (1595, 512)
├── test_script.npy         # Test text embeddings (6384, 512)
└── mean_pose.mat           # Mean pose for initialization
```

### Data Statistics

- Total samples: 31,919 motion-text pairs
- Train/Val/Test split: 80% / 5% / 15%
- Motion sequences: 32 timesteps, 24 dimensions (8 joints x 3 coordinates)
- Text embeddings: 512 dimensions (Universal Sentence Encoder)
- Motion data normalized to range [-1, 1]

## Repository Structure

```bash
.
├── config.py                      # Central configuration file
├── setup.py                       # Package installation script
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment file
├── .gitignore                     # Git ignore patterns
│
├── Data/                          # Training, validation, and test datasets
│
├── Models/                        # Trained model checkpoints
│   └── model_epoch_150.keras     # Final trained model
│
├── Results/                       # Generated outputs and visualizations
│
├── Notebooks/                     # Jupyter notebooks for analysis
│   ├── eval_model.ipynb          # Model evaluation
│   ├── evaluate_FGD.ipynb        # Frechet Gesture Distance
│   ├── test_GAN.ipynb            # GAN testing
│   └── ...
│
├── Hyperparameter_Tuning_and_Ablation_Study/
│   └── Model_and_Results/        # 20 experimental configurations
│       ├── Case_1/
│       ├── Case_2/
│       └── ...
│
├── src/                          # Main source code
│   ├── __init__.py
│   │
│   ├── Data/                     # Data preprocessing
│   │   ├── __init__.py
│   │   ├── preprocess_data.py   # Text embedding with USE
│   │   ├── dataset_split.py     # Train/val/test splitting
│   │   └── data_exploring.py    # Data inspection
│   │
│   ├── Train/                    # Training scripts
│   │   ├── __init__.py
│   │   ├── execute_model.py     # Main training script
│   │   └── structure_GAN.py     # Model architectures
│   │
│   ├── Evaluate/                 # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── calc_errors.py       # MAE and APE
│   │   ├── calc_jerk_or_acceleration.py  # Motion quality metrics
│   │   └── FGD/                 # Frechet Gesture Distance
│   │       ├── embedding_net.py
│   │       ├── train_AE.py
│   │       └── evaluate_FGD.py
│   │
│   ├── Validate/                 # Validation scripts
│   │   └── validate_GAN.py
│   │
│   ├── Visualisation/           # Visualization tools
│   │   ├── plotfunc.py          # 3D skeleton plotting
│   │   ├── test_GAN.py          # Generate visualizations
│   │   └── static_frame.py      # Static pose rendering
│   │
│   ├── Pepper_Implementation/   # Robot deployment
│   │   ├── publisher_py3.py     # Motion data server
│   │   ├── subscriber_py2.py    # Pepper robot client
│   │   └── pepper_rule_based.py # Inverse kinematics
│   │
│   └── utils/                   # Utility functions
│       └── my_functions.py      # Helper functions
│
└── job.slurm                    # HPC cluster job script
```

## Model Architecture

### Generator (LSTM-based)

- **Input**:
  - Noise vector (20-dim)
  - Text embedding (512-dim from Universal Sentence Encoder)
  - Initial pose (24-dim)
- **Architecture**:
  - 2 stacked LSTM layers (512 units each)
  - Dense output layer (24 units, tanh activation)
- **Output**: 32-frame gesture sequence (32 x 24)

### Discriminator (CNN-based)

- **Input**:
  - Motion sequence (32 x 24)
  - Text embedding (512-dim)
- **Architecture**:
  - 2 Conv1D layers (32 and 64 filters) with Spectral Normalization
  - Layer Normalization + ReLU activation
  - Global Average Pooling
  - Dense layers (128 units)
  - Dropout (0.2)
- **Output**: Real/Fake classification (sigmoid)

### Training Configuration

- Epochs: 150
- Batch size: 32
- Optimizer: Adam (learning rate: 0.00002 for both G and D)
- Loss components:
  - Adversarial loss (weight: 1)
  - Distance loss (weight: 10)
  - Velocity loss (weight: 5)
- Label smoothing: Real [0.9, 1.0], Fake [0.0, 0.1]

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **MAE (Mean Absolute Error)**: Overall motion similarity
- **APE (Average Position Error)**: Per-joint position accuracy
- **FGD (Frechet Gesture Distance)**: Distribution similarity in learned embedding space
- **Jerk**: Third derivative of position (smoothness)
- **Acceleration**: Second derivative of position
- **Velocity**: First derivative of position

## Usage Examples

### 1. Data Preprocessing

```python
from src.Data.preprocess_data import load_encoder
import config

# Load Universal Sentence Encoder
encoder = load_encoder(config.ENCODER_MODEL_PATH)

# Encode text
text = "A person is waving hello"
embedding = encoder([text]).numpy()
```

### 2. Training with Custom Configuration

```python
# Edit config.py
EPOCHS = 200
BATCH_SIZE = 64
ALPHA = 1.5

# Then run training
python src/Train/execute_model.py --case 15
```

### 3. Loading Trained Model

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('Models/model_epoch_150.keras')

# Generate motion
import numpy as np
noise = np.random.normal(0, 1, (1, 20))
text_embedding = encoder(["A person is gesturing"]).numpy()
initial_pose = np.load('Data/mean_pose.mat')

generated_motion = model([noise, text_embedding, initial_pose])
```

### 4. Visualization

```python
from src.Visualisation.plotfunc import plot3d_pose

# Visualize generated motion
plot3d_pose(generated_motion[0], save_path='output.gif')
```

### 5. Pepper Robot Deployment

```python
# Start the motion server
python src/Pepper_Implementation/publisher_py3.py

# On Pepper robot (Python 2.7 with NAOqi)
python src/Pepper_Implementation/subscriber_py2.py
```

## Hyperparameter Tuning

The repository includes 20 different experimental configurations in the `Hyperparameter_Tuning_and_Ablation_Study` directory. Each case explores different:

- Number of LSTM layers
- LSTM hidden units
- Loss weight combinations
- Learning rates
- Discriminator architectures

To run a specific case:

```bash
python execute_model.py --case 1  # Run Case 1
python execute_model.py --case 20 # Run Case 20
```

Results are automatically saved to:
```
Hyperparameter_Tuning_and_Ablation_Study/
├── Model_and_Results/Case_X/
│   ├── model_epoch_0.keras
│   ├── model_epoch_50.keras
│   ├── model_epoch_100.keras
│   ├── model_epoch_150.keras
│   └── lstm_gan_training_losses.png
└── Test_and_Generated_Data/Case_X/
    ├── Generated_Data/
    └── Test_Data/
```

## Code Quality Improvements

This repository follows Python best practices:

- **Modular structure**: All code organized into logical packages
- **Configuration management**: Centralized settings in `config.py`
- **Error handling**: Comprehensive try-except blocks with informative messages
- **Logging**: Professional logging instead of print statements
- **Type hints**: Clear function signatures with type annotations
- **Documentation**: Docstrings for all major functions
- **Command-line arguments**: Flexible execution without code editing

## GPU Support

The code automatically detects and uses available GPUs:

```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

For optimal performance:
- Single GPU: Automatically used
- Multiple GPUs: Uses data parallelism
- No GPU: Falls back to CPU (slower)

## Logging

Training progress is logged with detailed information:

```
2026-01-14 12:34:56 - __main__ - INFO - Starting LSTM-GAN training for Case 5
2026-01-14 12:34:56 - __main__ - INFO - Num GPUs Available: 1
2026-01-14 12:34:57 - __main__ - INFO - Loading training and validation data
2026-01-14 12:34:58 - __main__ - INFO - Train script shape: (25535, 512)
2026-01-14 12:34:58 - __main__ - INFO - Creating models
2026-01-14 12:35:00 - __main__ - INFO - Starting training
2026-01-14 12:35:05 - __main__ - INFO - Epoch 0/150 - Real loss: 0.6932, ...
```

## Acknowledgments

- University of Southampton for research support
- Universal Sentence Encoder by Google Research
- NAOqi SDK for Pepper robot integration
- The research community for gesture generation datasets

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: liangkj75@gmail.com
