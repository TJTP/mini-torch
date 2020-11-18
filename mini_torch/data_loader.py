import numpy as np
from math import ceil

class Batch:
    def __init__(self, inputs, labels, batch_len):
        self.inputs = inputs 
        self.labels = labels
        self.len = batch_len

class DataLoader:
    def __init__(self, features, labels, batch_size=8, shuffle=True):
        self._features = features
        self._labels = labels
        self._batch_size = batch_size
        self._shuffle = shuffle
        self.data_num = self._labels.shape[0]
        self.len = ceil(self.data_num / self._batch_size)
    
    def __call__(self):
        div_positions = np.arange(0, self.data_num, self._batch_size)

        if self._shuffle:
            idxs = np.arange(self.data_num)
            np.random.shuffle(idxs)
            self._features = self._features[idxs]
            self._labels = self._labels[idxs]
        
        for div_position in div_positions:
            end = div_position + self._batch_size
            batch_inputs = self._features[div_position: end]
            batch_labels = self._labels[div_position: end]
            yield Batch(inputs=batch_inputs, labels=batch_labels, batch_len=len(batch_labels))