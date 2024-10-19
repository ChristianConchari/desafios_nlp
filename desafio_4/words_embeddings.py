"""
This module contains the WordsEmbeddings class for handling word embeddings.
"""
import logging
from pathlib import Path
import pickle
import numpy as np

class WordsEmbeddings(object):
    """
    WordsEmbeddings class for handling word embeddings.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, embedding_dim, word_to_vec_path, pkl_path, word_max_size=60):
        self.word_to_vec_model_txt_path = word_to_vec_path
        self.pkl_path = pkl_path
        self.n_features = embedding_dim
        self.word_max_size = word_max_size
        self.default_embedding = np.zeros((self.n_features,), dtype=np.float32)  # Default for unknown words

        # Load the embeddings
        words_embedding_pkl = Path(self.pkl_path)
        if not words_embedding_pkl.is_file():
            words_embedding_txt = Path(self.word_to_vec_model_txt_path)
            assert words_embedding_txt.is_file(), 'Words embedding file not available.'
            embeddings = self.convert_model_to_pickle()
        else:
            embeddings = self.load_model_from_pickle()

        self.embeddings = embeddings
        # Build the vocabulary hashmap
        index = np.arange(self.embeddings.shape[0])
        # Dictionary mapping words to their indices
        self.word2idx = dict(zip(self.embeddings['word'], index))
        self.idx2word = dict(zip(index, self.embeddings['word']))

    def get_words_embeddings(self, words):
        """Get embeddings for a word or a list of words."""
        if isinstance(words, str):
            words = [words]
        words_idxs = self.words2idxs(words)
        return self.embeddings[words_idxs]['embedding']

    def words2idxs(self, words):
        """Convert a list of words to their corresponding indices."""
        return np.array([self.word2idx.get(word, -1) for word in words])

    def idxs2words(self, idxs):
        """Convert a list of indices back to their corresponding words."""
        return np.array([self.idx2word.get(idx, '-1') for idx in idxs])

    def load_model_from_pickle(self):
        """Load word embeddings from a pickle file."""
        self.logger.debug('Loading word embeddings from pickle: %s', self.pkl_path)

        try:
            with open(self.pkl_path, 'rb') as f_in:
                embeddings = pickle.load(f_in)
            self.logger.debug('Word embeddings loaded successfully.')
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings from {self.pkl_path}: {str(e)}") from e

    def convert_model_to_pickle(self):
        """Convert the word embeddings from the text format to a pickle file."""
        self.logger.debug('Converting and loading embeddings from text file: %s', self.word_to_vec_model_txt_path)
        structure = [('word', np.dtype('U' + str(self.word_max_size))), ('embedding', np.float32, (self.n_features,))]
        structure = np.dtype(structure)

        # Load embeddings from the text file
        with open(self.word_to_vec_model_txt_path, encoding="utf8") as words_embeddings_txt:
            embeddings_gen = (
                (line.split()[0], line.split()[1:])
                for line in words_embeddings_txt
                if len(line.split()[1:]) == self.n_features
            )
            embeddings = np.fromiter(embeddings_gen, structure, count=-1)

        # Add a null embedding
        null_embedding = np.array(
            [('null_embedding', np.zeros((self.n_features,), dtype=np.float32))],
            dtype=structure
        )
        embeddings = np.concatenate([embeddings, null_embedding])

        try:
            with open(self.pkl_path, 'wb') as f_out:
                pickle.dump(embeddings, f_out, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.debug('Word embeddings loaded and converted to pickle successfully.')
        except Exception as e:
            raise RuntimeError(f"Failed to save embeddings to {self.pkl_path}: {str(e)}") from e

        return embeddings


class GloveEmbeddings(WordsEmbeddings):
    """
    GloveEmbeddings class for handling GloVe word embeddings.
    """
    def __init__(self):
        super().__init__(
            embedding_dim=50,
            word_to_vec_path='glove.twitter.27B.50d.txt',
            pkl_path='gloveembedding.pkl',
            word_max_size=60
        )


class FasttextEmbeddings(WordsEmbeddings):
    """
    FasttextEmbeddings class for handling FastText word embeddings.
    """
    def __init__(self):
        super().__init__(
            embedding_dim=300,
            word_to_vec_path='crawl-300d-2M.vec',
            pkl_path='fasttext.pkl',
            word_max_size=60
        )
