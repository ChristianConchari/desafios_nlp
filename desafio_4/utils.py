"""
This module contains utility functions used in the project.
"""
import re
import contractions
import numpy as np

def clean_text(txt):
    """
    Cleans the input text by performing the following operations:
        1. Expands contractions (e.g., "don't" -> "do not").
        2. Converts the text to lowercase.
        3. Removes numbers and special characters, retaining only letters and spaces.
        4. Removes extra whitespaces.
    """
    # Expand contractions
    txt = contractions.fix(txt)

    # Convert text to lowercase
    txt = txt.lower()

    # Remove numbers and special characters (keep punctuation for tokenizer)
    txt = re.sub(r'[^a-zA-Z\s]', '', txt)

    # Remove extra whitespaces
    txt = re.sub(r'\s+', ' ', txt).strip()

    return txt

def generate_answer(input_seq, encoder_model, decoder_model, max_out_len, word2idx_outputs, idx2word_target):
    """
    Generates an answer for the input sequence using the encoder and decoder models.
    """
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']

    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        print("Predicted index:", idx)  # Print the predicted index to debug

        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        states_value = [h, c]
        target_seq[0, 0] = idx

    return ' '.join(output_sentence)
