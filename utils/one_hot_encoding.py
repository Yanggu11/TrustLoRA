import numpy as np


class OneHotEncoder:
    def __init__(self, vocab=None):
        """
        Args:
            vocab (list, optional): List of unique classes. If None, it will be built from data.
        """
        self.vocab = vocab
        self.class_to_id = {}
        self.id_to_class = {}

        if vocab is not None:
            self.build_vocab(vocab)

    def build_vocab(self, vocab):
        self.vocab = list(vocab)
        self.class_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_class = {i: token for i, token in enumerate(vocab)}

    def fit(self, data):
        """Build vocabulary from data"""
        unique_classes = sorted(set(data))
        self.build_vocab(unique_classes)

    def encode(self, data):
        """Returns one-hot encoded numpy array"""
        if self.vocab is None:
            raise ValueError(
                "Vocabulary is not built. Call fit() or provide vocab during initialization."
            )

        one_hot = np.zeros((len(data), len(self.vocab)), dtype=int)
        for i, token in enumerate(data):
            if token in self.class_to_id:
                one_hot[i, self.class_to_id[token]] = 1
            else:
                raise ValueError(f"Token '{token}' not in vocabulary.")
        return one_hot

    def decode(self, one_hot):
        if self.id_to_class is None:
            raise ValueError(
                "Vocabulary is not built. Call fit() or provide vocab during initialization."
            )
        return [self.id_to_class[np.argmax(row)] for row in one_hot]
