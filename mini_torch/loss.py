
"""
损失函数
目前实现了
    (平均)平方损失
    对数似然损失函数 (log-likehood loss function)
    对数似然代价函数/softmax交叉熵损失函数
"""
from mini_torch.tensor import log, exp 
import numpy as np 

class SquareLoss:
    def loss(self, predicted, real):
        n = predicted.shape[0]
        loss_sum = 1/2 * ((predicted - real) * (predicted - real)).sum()

        return loss_sum / n

class LogLikehoodLoss:
    """
    用于采用sigmoid输出的二分类任务
    """
    def loss(self, predicted, real):
        losses = -real * log(predicted) - (1 - real) * log((1 - predicted))
        return losses.sum()

class SoftmaxCrossEntropyLoss:
    """
    用于多分类的softmax
    """
    def loss(self, predicted, real):
        exps = exp(predicted)
        probs = exps / exps.sum(axis=1, keepdims=True)
        loss_each = -log((probs * real).sum(axis=1, keepdims=True))
        return loss_each.sum() / real.shape[0] # 这个地方的sum里面不能加上axis参数, 因为要得到一个标量而不是向量