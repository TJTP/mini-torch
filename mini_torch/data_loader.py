import numpy as np
from collections import namedtuple

Batch = namedtuple("Batch", ["inputs", "labels"])

class DataLoader():
    def __init__(self, batch_size=8, shuffle=True):
        self._batch_size = batch_size
        self._shuffle = shuffle
    
    def __call__(self, inputs, labels):
        inputs_num = len(inputs)
        div_positions = np.arange(0, inputs_num, self._batch_size)

        if self._shuffle:
            idxs = np.arange(inputs_num)
            np.random.shuffle(idxs)
            inputs = inputs[idxs]
            labels = labels[idxs]
        
        for div_position in div_positions:
            end = div_position + self._batch_size
            batch_inputs = inputs[div_position: end]
            batch_labels = labels[div_position: end]
            yield Batch(inputs=batch_inputs, labels=batch_labels)
