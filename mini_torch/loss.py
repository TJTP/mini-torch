import numpy as np 
import mini_torch.operation as ops 

class BaseLoss():
    def loss(self, predicted, real):
        raise NotImplementedError

class MeanSquareLoss(BaseLoss):
    def loss(self, predicted, real):
        #n = predicted.shape[0]
        loss_square = ops.pow(predicted - real, 2)
        loss_sum = 1 / 2 * loss_square.sum()

        return loss_sum

