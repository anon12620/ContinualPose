import torch
from mmpose.registry import HOOKS

from .snapshot_plugin import SnapshotPlugin


def load_state_dict(model, state_dict, strict=False, logger=None):
    """
    Load parameters from a state dictionary into a model, padding with zeros if necessary.
    
    Args:
        model (torch.nn.Module): The model to which the state_dict will be loaded.
        state_dict (dict): State dictionary from the previous model.
        strict (bool): If True, ensures that the keys in state_dict and model match exactly.
        logger (Logger): Optional logger for logging information about the loading process.
    """
    own_state = model.state_dict()
    unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    for name, param in state_dict.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                resized_param = torch.zeros_like(own_state[name])
                slices = tuple(slice(0, min(dim_model, dim_pretrained)) for dim_model, dim_pretrained in zip(own_state[name].shape, param.shape))
                resized_param[slices] = param[slices]
                own_state[name].copy_(resized_param)
                if logger:
                    logger.info(f'{name}: Loaded weights with adjustment for shape mismatch.')
        else:
            if strict:
                raise RuntimeError(f'Error: {name} is an unexpected key in the state_dict.')
            if logger:
                logger.warning(f'Warning: {name} not found in the model and will be ignored.')

    if missing_keys:
        error_msg = f'Missing keys in model state_dict: {", ".join(missing_keys)}.'
        if strict:
            raise RuntimeError(error_msg)
        if logger:
            logger.warning(error_msg)

    if unexpected_keys:
        error_msg = f'Unexpected keys in state_dict: {", ".join(unexpected_keys)}.'
        if strict:
            raise RuntimeError(error_msg)
        if logger:
            logger.warning(error_msg)


@HOOKS.register_module()
class DynamicSnapshotPlugin(SnapshotPlugin):
    def __init__(self, model_cfgs: dict = {}, mode="last", copy_weights=True):
        """
        A plugin for dynamically managing model snapshots and adapting the model
        architecture based on new tasks or experiences.

        This plugin extends SnapshotPlugin to not only preserve snapshots of the model
        at various stages but also adapt the model to new requirements after each
        experience. The dynamic aspect emphasizes its ability to adjust the architecture
        in a flexible manner, making it suitable for continual learning scenarios where
        model evolution is necessary.

        Args:
            mode (str): Determines the snapshot retention strategy; can be 'last' for
                        keeping only the most recent snapshot or 'all' for keeping all
                        snapshots.
        """
        super().__init__(mode=mode)
        assert isinstance(model_cfgs, dict), \
            "model_cfgs should be a dictionary with experience indices as keys and \
            model configurations as values."
        self.model_cfgs = model_cfgs
        self.copy_weights = copy_weights

    def switch_model(self, runner, experience_index):
        """
        Abstract method to adapt the model's architecture. Should be implemented by subclasses
        to specify how the model should be modified in response to new experiences.

        Args:
            runner: The object managing the model and training process.
            experience_index: The index of the current experience in a continual learning scenario.
        """
        # Get configuration for the next experience
        next_exp = experience_index + 1
        cfg = self.model_cfgs.get(next_exp, None)
        if cfg is None:
            runner.logger.info(f"[DynamicSnapshotPlugin] M_{next_exp} is same as  M_{experience_index}")
            return

        # Update the model configuration
        runner.logger.info(f"[DynamicSnapshotPlugin] Switching model for E_{next_exp}")
        runner.switch_model(cfg)

        # Load weights from the previous model (if available)
        last_state_dict = runner.last_state_dict
        if self.copy_weights and last_state_dict is not None:
            runner.logger.info(f"[DynamicSnapshotPlugin] Initializing M_{next_exp} with M_{experience_index} weights")
            load_state_dict(runner.module, last_state_dict, 
                            strict=False, logger=runner.logger)

    def after_experience(self, runner, experience_index):
        """
        Hook to be called after each experience. It saves the snapshot of the model and
        potentially adapts its architecture according to the implementation of switch_model.

        Args:
            runner: The object managing the model and training process.
            experience_index: The index of the current experience in a continual learning scenario.
        """
        super().after_experience(runner, experience_index)
        self.switch_model(runner, experience_index)
