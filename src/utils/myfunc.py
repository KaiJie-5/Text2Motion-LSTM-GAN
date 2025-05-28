import re
import tensorflow_hub as hub

# Remove special characters\whitespaces
def normalize_text(text):
    text = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")]
    return text

def load_encoder(model_path):
    #Output's dimension = 512
    encoder = hub.load(model_path)
    return encoder



