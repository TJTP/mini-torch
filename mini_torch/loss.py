
"""
损失函数
目前只实现了(平均)平方损失
"""

class SquareLoss:
    def loss(self, predicted, real):
        n = predicted.shape[0]
        loss_sum = 1/2 * ((predicted - real) * (predicted - real)).sum()

        return loss_sum / n

