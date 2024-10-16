import re
import numpy as np
from tensorflow.keras.utils import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    """
    Learning rate schedule that decays by 5% every epoch.
    """
    return 1e-3 * (0.95 ** epoch)  # Decay the learning rate gradually

lr_scheduler = LearningRateScheduler(lr_schedule)

def perplexity(y_true, y_pred):
    """
    Compute the perplexity of a model given the true and predicted labels.
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity_val = K.exp(K.mean(cross_entropy))
    return perplexity_val

def pad_and_concatenate_sentences(tokenized_sentences, max_context_size):
    """
    Apply padding to sentences so that all have the same length and concatenate them into a single tensor.
    """
    padded_sentences = []
    for sent in tokenized_sentences:
        subseq = [sent[:i+2] for i in range(len(sent)-1)]
        padded_sentences.append(pad_sequences(subseq, maxlen=max_context_size+1, padding='pre'))
    return np.concatenate(padded_sentences, axis=0)

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing numbers and special characters, and removing extra spaces.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Delete numbers
    text = re.sub(r'[^\w\s\']', '', text)  # Keep only words and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def generate_seq(model, tokenizer, seed_text, max_length, n_words):
    """
    Generates a sequence of words based on the seed text using the trained model.
    """
    output_text = seed_text

    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([output_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        predicted_word_idx = np.argmax(model.predict(encoded), axis=-1)[0]
        out_word = tokenizer.index_word.get(predicted_word_idx, '')

        if out_word == '':
            break

        output_text += ' ' + out_word

    return output_text
