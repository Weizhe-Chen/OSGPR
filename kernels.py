import torch
from torch.nn.parameter import Parameter

import utils

constraint = utils.GreaterThanConstraint(min_value=0.0)


class GaussianKernel(torch.nn.Module):
    def __init__(self, length_scale, output_scale):
        super().__init__()
        self.free_length_scale = Parameter(
            constraint.inv_transform(utils.to_tensor(length_scale))
        )
        self.free_output_scale = Parameter(
            constraint.inv_transform(utils.to_tensor(output_scale))
        )

    @property
    def length_scale(self):
        return constraint.transform(self.free_length_scale)

    @length_scale.setter
    def length_scale(self, value):
        self.free_length_scale = Parameter(
            constraint.inv_transform(utils.to_tensor(value))
        )

    @property
    def output_scale(self):
        return constraint.transform(self.free_output_scale)

    @output_scale.setter
    def output_scale(self, value):
        self.free_output_scale = Parameter(
            constraint.inv_transform(utils.to_tensor(value))
        )

    def forward(self, x1, x2=None, diag=False):
        if diag:
            return self.output_scale * torch.ones(x1.shape[0], 1).to(x1)
        if x2 is None:
            x2 = x1
        x1 = x1 / self.length_scale
        x2 = x2 / self.length_scale
        dist = torch.cdist(x1, x2, p=2)
        return self.output_scale * torch.exp(-0.5 * dist.square())
