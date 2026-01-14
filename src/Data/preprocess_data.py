import tensorflow_hub as hub
import os
import scipy.io as scio
import numpy as np

model_path = '../2'

def load_encoder(model_path):
    """
    Load Universal Sentence Encoder model.
    Output dimension is 512.
    """
    encoder = hub.load(model_path)
    return encoder

encoder = load_encoder(model_path)
