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
├── Data/                               # Training, validation, and test datasets (.npy, .mat, .npz)
├── Hyperparameter_Tuning_and_Ablation_Study/  # Configurations and results for ablation and tuning experiments
├── Models/                             # Trained model checkpoints (.keras)
├── Notebooks/                          # Jupyter notebooks for training, evaluation, and visualization
├── Results/                            # Qualitative results and visual outputs
├── .gitattributes                      # Git LFS tracking for large files
