import torch
from torch import Tensor
from torch.nn import Module


class DebugActivationLayer(Module):
    """
    identity function returning input as ouput,
    records active neurones
    the metric used is the area above curve
    """
    def __init__(self):
        super(DebugActivationLayer, self).__init__()
        self.n_games = 0
        self.shape_to_last_active_mat = {}
        self.shape_to_result = {}
        self.max_ndim = -1

    def forward(self, input: Tensor) -> Tensor:
        input_ndim = input.ndim
        if input_ndim < self.max_ndim:
            return input
        else:
            self.max_ndim = input_ndim

        self.n_games += 1
        with torch.no_grad():
            mask = input.max(0).values.squeeze() > 0
            last_active_mat = self.shape_to_last_active_mat.setdefault(input_ndim, torch.zeros(mask.shape))
            last_active_mat[mask] = self.n_games
            total_area = self.n_games * mask.nelement()
            non_active_neurons_area_above_curve = (total_area - last_active_mat.sum()) / total_area
            self.shape_to_result[input_ndim] = non_active_neurons_area_above_curve.item()

        return input

    def get_non_active_neurons_area_above_curve(self):
        return self.shape_to_result[self.max_ndim]
