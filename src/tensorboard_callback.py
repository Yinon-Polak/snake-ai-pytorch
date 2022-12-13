import warnings

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_getattr


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count < 1_000:
            return True

        state_dicts_names, _ = self.model._get_torch_save_params()
        attr_map = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self.model, name)
            attr_map[name] = attr

        lr = attr_map['policy'].optimizer.state_dict()['param_groups'][0]['lr']
        pretty_params_names = list(attr_map['policy'].features_extractor.state_dict().keys())
        parameters = list(attr_map['policy'].features_extractor.parameters())

        with torch.no_grad():
            for pretty_name, p in zip(pretty_params_names, parameters):
                update_2_data_ratio = ((lr * p.grad).std() / p.data.std()).log10().item()
                self.logger.record(f"ud-{pretty_name}", update_2_data_ratio)

        return True

    # def _on_rollout_end(self) -> bool:
    #     self.step_count += 1
    #     if self.step_count < 100:
    #         return True
    #
    #     state_dicts_names, _ = self.model._get_torch_save_params()
    #     attr_map = {}
    #     for name in state_dicts_names:
    #         attr = recursive_getattr(self.model, name)
    #         attr_map[name] = attr
    #
    #     lr = attr_map['policy'].optimizer.state_dict()['param_groups'][0]['lr']
    #     pretty_params_names = list(attr_map['policy'].features_extractor.state_dict().keys())
    #     parameters = list(attr_map['policy'].features_extractor.parameters())
    #
    #     with torch.no_grad():
    #         for pretty_name, p in zip(pretty_params_names, parameters):
    #             update_2_data_ratio = ((lr * p.grad).std() / p.data.std()).log10().item()
    #             self.logger.record(f"ud-{pretty_name}", update_2_data_ratio)
    #
    #     return True
