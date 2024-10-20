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

def generate_answer_with_attention(input_seq, encoder_model, decoder_model, max_out_len, word2idx_outputs, idx2word_target):
    """
    Generates an answer for the input sequence using the encoder and decoder models with attention mechanism.
    """
    # Encode the input as state vectors and the encoder outputs
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    states_value = [state_h, state_c]

    # Initialize the target sequence with the start token '<sos>'
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']

    eos = word2idx_outputs['<eos>']
    output_sentence = []

    # Start generating the answer by iterating up to max_out_len
    for _ in range(max_out_len):
        # Predict the next word and the updated states with attention
        # Pass the current target sequence, encoder outputs, and decoder states
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs] + states_value)

        # Get the index of the predicted word
        idx = np.argmax(output_tokens[0, 0, :])

        print("Predicted index:", idx)

        # If the predicted index is <eos>, stop generating
        if eos == idx:
            break

        # Append the predicted word to the output sentence
        word = ''
        if idx > 0:
            word = idx2word_target.get(idx, '<unk>')  # Handle unknown words gracefully
            output_sentence.append(word)

        # Update the target sequence and the states
        target_seq[0, 0] = idx
        states_value = [h, c] 

    return ' '.join(output_sentence)