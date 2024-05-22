import copy

from mmpose.registry import HOOKS

from .base import BasePlugin
from ..utils import freeze_everything


@HOOKS.register_module()
class SnapshotPlugin(BasePlugin):
    def __init__(self, mode="last"):
        """
        Base class for plugins that require snapshots of the model.

        This plugin is used to store a copy of the model after each experience,
        which can be used for distillation or other purposes.
        """
        super().__init__()
        assert mode in ["last", "all"]
        self.mode = mode

    def _copy_model(self, runner):
        prev_model = copy.deepcopy(runner.module)
        state_dict = runner.module.state_dict()
        freeze_everything(prev_model)
        return prev_model, state_dict

    def _save_snapshot(self, runner):
        assert self.mode == "last"
        assert isinstance(runner.memory, dict) or runner.memory is None
        if runner.memory is None:
            runner.memory = dict()

        model, state_dict = self._copy_model(runner)
        runner.memory["model"] = model
        runner.memory["state_dict"] = state_dict

    def append_snapshot(self, runner):
        assert self.mode == "all"
        assert isinstance(runner.memory, dict) or runner.memory is None
        if runner.memory is None:
            runner.memory = dict()

        if "model" not in runner.memory:
            runner.memory["model"] = []
        
        if "state_dict" not in runner.memory:
            runner.memory["state_dict"] = []

        model, state_dict = self._copy_model(runner)
        runner.memory["model"].append(model)
        runner.memory["state_dict"].append(state_dict)

    def save_snapshot(self, runner):
        if self.mode == "last":
            self._save_snapshot(runner)
        elif self.mode == "all":
            self.append_snapshot(runner)
        else:
            raise ValueError("Invalid mode")
        
        history = runner.memory.get("model", [])
        if not isinstance(history, list):
            history = [history]

    def after_experience(self, runner, experience_index):
        """
        Save a copy of the model after each experience.
        """
        super().after_experience(runner, experience_index)
        self.save_snapshot(runner)
        
