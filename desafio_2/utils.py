import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from typing import Tuple

def reduce_dimensions(model: Word2Vec, num_dimensions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduces the dimensionality of word vectors using t-SNE.

    Parameters:
    model (gensim.models.Word2Vec): The Word2Vec model containing word vectors.
    num_dimensions (int, optional): The number of dimensions to reduce to. Default is 2.

    Returns:
    tuple: A tuple containing:
        - vectors (numpy.ndarray): The reduced-dimensionality vectors.
        - labels (numpy.ndarray): The labels corresponding to the vectors.
    """
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)

    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    return vectors, labels

class callback(CallbackAny2Vec):
    """
    Callback class to print the training loss after each epoch.

    This class inherits from `CallbackAny2Vec` and overrides the `on_epoch_end` method
    to print the loss after each epoch during the training of a Word2Vec model.

    Attributes:
        epoch (int): The current epoch number, initialized to 0.
        loss_previous_step (float): The loss value from the previous epoch, used to calculate the difference in loss.

    Methods:
        __init__():
            Initializes the callback with the starting epoch set to 0.
        on_epoch_end(model):
            Called at the end of each epoch. Prints the loss for the current epoch.
            If it's not the first epoch, it also prints the difference in loss from the previous epoch.
    """
    
    def __init__(self):
        """
        Initializes the callback with the starting epoch set to 0.
        """
        self.epoch = 0

    def on_epoch_end(self, model):
        """
        Called at the end of each epoch. Prints the loss for the current epoch.
        If it's not the first epoch, it also prints the difference in loss from the previous epoch.

        Parameters:
            model (gensim.models.Word2Vec): The Word2Vec model being trained.
        """
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
            
        self.epoch += 1
        self.loss_previous_step = loss
        