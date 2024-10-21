"""
This module contains utility functions
"""
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import InputExample, InputFeatures

def plot_train_metrics(history):
    """
    Plot the training metrics
    """
    epoch_count = range(1, len(history.history['accuracy']) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    sns.lineplot(x=epoch_count, y=history.history['accuracy'], label='train', ax=ax1)
    sns.lineplot(x=epoch_count, y=history.history['val_accuracy'], label='valid', ax=ax1)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)  # Add grid

    # Plot loss
    sns.lineplot(x=epoch_count, y=history.history['loss'], label='train', ax=ax2)
    sns.lineplot(x=epoch_count, y=history.history['val_loss'], label='valid', ax=ax2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)  # Add grid

    plt.show()


def download_file(url, filename):
    """
    Download a file from a URL if it does not already exist.
    """
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True, timeout=10)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

def convert_data_to_examples(text, labels, idx_train, idx_val):
    """
    Convert text and labels into InputExample objects for training and validation.
    """
    # Create InputExample objects for training data
    train_input_examples = [
        InputExample(
            guid=None,  # Globally unique ID for bookkeeping, unused in this case
            text_a=text[i],  # The text data
            text_b=None,  # No second text segment
            label=labels[i]  # The label corresponding to the text
        ) for i in idx_train
    ]

    # Create InputExample objects for validation data
    validation_input_examples = [
        InputExample(
            guid=None,  # Globally unique ID for bookkeeping, unused in this case
            text_a=text[i],  # The text data
            text_b=None,  # No second text segment
            label=labels[i]  # The label corresponding to the text
        ) for i in idx_val
    ]

    return train_input_examples, validation_input_examples

def convert_examples_to_tf_dataset(examples, bert_tokenizer, output_shape=3, max_length=512):
    """
    Convert InputExample objects to a TensorFlow Dataset for BERT model training.
    """
    def gen():
        """
        Generator function to yield features one by one.
        """
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                },
                f.label,
            )

    # List to store the features
    features = []

    for e in examples:
        input_dict = bert_tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=max_length,  # Truncate sequences longer than max_length
            return_token_type_ids=False,  # Do not return token type IDs
            return_attention_mask=True,  # Return attention mask
            padding='max_length',  # Pad sequences to max_length
            truncation=True  # Truncate sequences longer than max_length
        )

        # Extract input_ids and attention_mask from the encoded dictionary
        input_ids, attention_mask = input_dict["input_ids"], input_dict["attention_mask"]

        # Create an InputFeatures object and append it to the features list
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label=e.label
            )
        )

    # Create a TensorFlow Dataset from the generator
    return tf.data.Dataset.from_generator(
        gen,
        (
            {"input_ids": tf.int32, "attention_mask": tf.int32},  # Output types
            tf.float32  # Label type
        ),
        (
            {
                "input_ids": tf.TensorShape([None]),  # Shape of input_ids
                "attention_mask": tf.TensorShape([None]),  # Shape of attention_mask
            },
            tf.TensorShape([output_shape]),  # Shape of labels
        ),
    )
