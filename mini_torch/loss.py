
"""
损失函数
目前实现了
    (平均)平方损失
    对数似然损失函数 (log-likehood loss function)
    对数似然代价函数/softmax交叉熵损失函数
"""
from mini_torch.tensor import log, exp 

class SquareLoss:
    def __init__(self):
        self.name = "Square loss"

    def loss(self, logits, labels):
        n = labels.shape[0]
        loss_sum = 1/2 * ((logits - labels) * (logits - labels)).sum()

        return loss_sum / n

class LogLikehoodLoss:
    """
    用于采用sigmoid输出的二分类任务
    """
    def __init__(self):
        self.name = "Log likehood loss"

    def loss(self, logits, labels):
        n = labels.shape[0]
        losses = -labels * log(logits) - (1 - labels) * log((1 - logits))
        return losses.sum() / n

class SoftmaxCrossEntropyLoss:
    """
    用于多分类的softmax
    """
    def __init__(self):
        self.name = "Softmax cross entropy loss"

    def loss(self, logits, labels):
        n = labels.shape[0]
        exps = exp(logits)
        probs = exps / exps.sum(axis=1, keepdims=True)
        loss_each = -log((probs * labels).sum(axis=1, keepdims=True))
        return loss_each.sum() / n # 这个地方的sum里面不能加上axis参数, 因为要得到一个标量而不是向量