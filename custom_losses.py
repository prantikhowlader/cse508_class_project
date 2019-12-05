import torch
import torch.nn as nn


def L2_normsq(tensor):
    loss = (tensor**2).mean()
    return loss


class TotalVaryLoss(nn.Module):
    def __init__(self):
        super(TotalVaryLoss, self).__init__()

    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        w.requires_grad_(False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
                                         torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss