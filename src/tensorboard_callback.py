import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_getattr

from src.utils.model.debug_activation_layer import DebugActivationLayer


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
        wandb_logs = {}
        with torch.no_grad():
            for pretty_name, p in zip(pretty_params_names, parameters):
                update_2_data_ratio = ((lr * p.grad).std() / p.data.std()).log10().item()
                self.logger.record(f"ud-{pretty_name}", update_2_data_ratio)
                wandb_logs[f"ud-{pretty_name}"] = update_2_data_ratio

        for upper_layer_name, layer in attr_map['policy'].features_extractor.named_children():
            # if upper_layer_name == 'cnn' and isinstance(layer, nn.Sequential):
            debug_layers = filter(lambda x: isinstance(x[1], DebugActivationLayer), layer.named_children())
            for layer_index, debug_layer in debug_layers:
                title = f"AAC non-active neurons {layer_index}"
                aac_non_active_neurons = debug_layer.get_non_active_neurons_area_above_curve()
                self.logger.record(title, aac_non_active_neurons)
                wandb_logs[title] = aac_non_active_neurons

        wandb.log(wandb_logs)
        return True
