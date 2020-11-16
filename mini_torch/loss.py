import numpy as np 
import mini_torch.operation as ops 

class BaseLoss():
    def loss(self, predicted, real):
        raise NotImplementedError

class SquareLoss(BaseLoss):
    def loss(self, predicted, real):
        #n = predicted.shape[0]
        loss_square = ops.pow(predicted - real, 2)
        loss_sum = loss_square.sum()

        return loss_sum

