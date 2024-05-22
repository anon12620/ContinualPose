from mmengine.hooks import Hook
from mmpose.registry import HOOKS

from typing import Optional, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class BasePlugin(Hook):
    def __init__(self):
        super().__init__()

    def before_first_experience(self, runner):
        runner.logger.info(f"Loading {self.__class__.__name__}")

    def before_experience(self, runner, experience_index: int):
        pass

    def after_experience(self, runner, experience_index: int):
        pass

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        pass

    def after_backward(self, runner, batch_idx: int, data_batch: DATA_BATCH = None):
        pass
