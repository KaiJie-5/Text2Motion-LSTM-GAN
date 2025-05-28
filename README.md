# Robot Gesture Generation using LSTM-GAN

This repository contains the implementation of an LSTM-based Generative Adversarial Network (GAN) for synthesizing human-like gestures from text inputs. The project is part of my final year individual research at the University of Southampton, focusing on communicative robot motion generation.

## Project Overview

The goal of this project is to generate realistic gesture sequences conditioned on textual descriptions. The model architecture consists of:
- **Text Encoder**: Embeds and processes input text using LSTM.
- **Generator (LSTM-GAN)**: Produces gesture sequences from noise and encoded text.
- **Discriminator**: Evaluates the realism of generated sequences and distinguishes between real and synthetic motion.

## Repository Structure

```bash
.
├── Data/                          # Training, validation, and test datasets
│   ├── pose/                      # Raw 3D pose sequences
│   ├── script/                    # Text scripts aligned with poses
│   ├── mean_pose.mat              # Mean pose for normalization
│   ├── metadata.npz              # Dataset metadata (e.g., labels, timestamps)
│   ├── total_script.txt          # Raw textual descriptions
│   ├── train_action.npy          # Training motion data
│   ├── val_action.npy            # Validation motion data
│   ├── test_action.npy           # Test motion data
│   ├── train_script.npy          # Text embeddings for training set
│   ├── val_script.npy            # Text embeddings for validation set
│   └── test_script.npy           # Text embeddings for test set
│
├── Models/                        # Trained model checkpoints (.keras)
│   └── model_epoch_150.keras     # Final model after 150 epochs
│
├── Notebooks/                    # Jupyter notebooks for testing and analysis
│   ├── eval_model.ipynb          # Evaluation (MAE, jerk, velocity)
│   ├── evaluate_FGD.ipynb        # Fréchet Gesture Distance calculation
│   ├── feat_extraction.ipynb     # Feature extraction
│   ├── MModality_generation.ipynb# Multimodality score calculation
│   ├── static_frame.ipynb        # Static pose visualization
│   ├── test_GAN.ipynb            # GAN inference and testing
│   └── testing.ipynb             # General testing scripts
│
├── Results/                      # Visual outputs and best model results
│
├── src/                          # Main codebase
│   ├── Data/                     # Data loading and preprocessing
│   ├── Evaluate/                 # Evaluation metric implementations
│   ├── Pepper_Implementation/   # Code for deploying on Pepper robot
│   ├── Train/                    # Training scripts for LSTM-GAN
│   ├── utils/                    # Helper functions (e.g., inverse kinematics)
│   ├── Validate/                 # Validation workflows
│   └── Visualisation/           # Animation and motion rendering tools
│
├── Hyperparameter_Tuning_and_Ablation_Study/  
│                                 # Config files and results for ablation experiments
│
├── environment.yml              # Conda environment definition
├── job.slurm                    # SLURM job script for HPC cluster training
└── .gitattributes               # Git LFS tracking for large files
