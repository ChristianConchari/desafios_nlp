"""
This module contains utility functions for analyzing the similarity of documents and terms.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_similarity(
    most_similar_idx: list,
    documents: list,
    labels: list,
    class_names: list,
    cosine_sim_matrix: np.ndarray,
    main_idx: int,
    top_n: int = 5
    ):
    """
    Analyze the similarity of selected documents with the rest of the documents.

    Parameters:
    - most_similar_idx: List of indices of the most similar documents.
    - documents: List of documents in the dataset.
    - labels: List of labels corresponding to each document.
    - class_names: List of class names corresponding to each label.
    - cosine_sim_matrix: Cosine similarity matrix of the documents.
    - main_idx: Index of the main document to analyze.
    - top_n: Number of most similar documents to retrieve for each selected document.
    """
    
    print(f"\nTop {top_n} Similar Documents:\n{'-'*50}")

    for idx in most_similar_idx:
        print(f"\n{'='*50}")
        print(f"Analyzing Document {idx}")
        print(f"{'='*50}")
        print(f"Content:\n{documents[idx][:200]}\n")
        print(f"Document Class: {class_names[labels[idx]]}")
        print(f"Main Document Class: {class_names[labels[main_idx]]}")
        print(f"{'='*50}")

def get_top_similar_words(term, matrix, terms, top_n=5):
    """
    Get the top similar words to a given term based on a similarity matrix.

    Parameters:
    - term (str): The term to find similar words for.
    - matrix (numpy.ndarray): The similarity matrix.
    - terms (list): The list of terms corresponding to the rows/columns of the matrix.
    - top_n (int): The number of top similar words to return. Default is 5.

    Returns:
    - similar_words (list): A list of tuples containing the similar words and their similarity scores.
    """
    # Get the index of the term in the terms list
    idx = terms.tolist().index(term) 
    
    # Get the term vector
    term_vector = matrix[idx].toarray() 
    
    similarities = cosine_similarity(term_vector, matrix) 
    similar_indices = np.argsort(similarities[0])[::-1][1:top_n+1] 
    similar_words = [(terms[i], similarities[0][i]) for i in similar_indices]
    return similar_words
