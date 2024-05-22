from typing import Dict, List, Tuple, Union

import torch
from tqdm import tqdm
from mmpose.registry import HOOKS

from .lwf import LWFPlugin


@HOOKS.register_module()
class LGFPlugin(LWFPlugin):
    """Learning Gradually without Forgetting (LGF) Plugin.

    LGF extends the Learning without Forgetting (LwF) algorithm with a gradual
    unfreezing strategy. The importance of each layer is computed based on the
    sum of the squared gradients of all parameters within the layer. The
    importance values are normalized across layers and used to determine the
    epoch at which each layer should be unfrozen.

    This ensures that the least important layers are fine-tuned first, while the
    most important layers are fine-tuned later in the training process. This
    strategy can help to prevent catastrophic forgetting and improve the
    plasticity of the model.

    Args:
        max_epochs (int): Maximum number of epochs for training.
        temperature (float): Temperature for distillation loss.
        lambda_lgf (float or list[float]): Weight for distillation loss.
            If a list is provided, it should have the same length as the number
            of experiences. If a single value is provided, it will be used for
            all experiences.
        converters (Dict[Dict]): Dictionary with experience index as key and
            'KeypointConverter' configuration as value. This converter will be
            used to convert the model predictions at the corresponding experience
            index to the format of the model at the previous experience index.
            This is only necessary if the model architecture changes between
            experiences, i.e. when the number of keypoints or the keypoint order
            changes. If the model architecture is the same for all experiences,
            this argument can be omitted.
        unfreeze_schedule (Dict): A dictionary where keys are epoch numbers and
            values are lists of component names to unfreeze.
    """

    def __init__(
            self,
            max_epochs: Union[int, List[int]],
            temperature: float = 2.0,
            lambda_lgf: float = 0.5,
            converters: Dict[int, Dict] = None,
            scale_lwf: bool = False,
            unfreeze_schedule: Dict[int, List[str]] = None,
    ):
        super().__init__(temperature, lambda_lgf, converters)
        assert unfreeze_schedule is None or isinstance(unfreeze_schedule, dict)
        self.max_epochs = max_epochs  # maximum number of epochs for training
        self.scale_lwf = scale_lwf  # whether to scale the LwF loss
        self.unfreeze_schedule = unfreeze_schedule or dict()
        self.frozen_modules = set()  # set of frozen modules
        self.experience_id = -1  # Current experience
        self.prev_dataloader = None  # Dataloader from the previous experience

    def before_experience(self, runner, experience_index: int):
        super().before_experience(runner, experience_index)
        self.experience_id = experience_index

    def set_requires_grad(self, model, component_names, requires_grad):
        """
        Set requires_grad for the parameters of specified components, supporting nested components.
        """

        def _set_requires_grad_for_module(module, name_prefix=''):
            """
            Recursively set requires_grad for parameters in the module and its children.
            """
            for name, child in module.named_children():
                # Construct the full name of the current module
                full_name = f"{name_prefix}.{name}" if name_prefix else name

                # Check if the current module or any of its children is in the target list
                if any(full_name.startswith(component_name) for component_name in component_names):
                    for param in child.parameters():
                        param.requires_grad = requires_grad

                    # Update the set of frozen/unfrozen components accordingly
                    if requires_grad:
                        self.frozen_modules.discard(full_name)
                    else:
                        self.frozen_modules.add(full_name)

                # Recursively apply to child modules
                _set_requires_grad_for_module(child, full_name)

        # Start the recursive process from the top-level model
        _set_requires_grad_for_module(model)

        # Ensure the top level is also checked if it's in the component_names
        if "" in component_names:
            for param in model.parameters():
                param.requires_grad = requires_grad
            if requires_grad:
                self.frozen_modules.discard("top-level model")
            else:
                self.frozen_modules.add("top-level model")

    def before_train_epoch(self, runner):
        """Handle the freezing and unfreezing of modules at specified epochs."""
        epoch = runner.epoch
        model = runner.module

        # Get unfreeze schedule for current experience (if any)
        unfreeze_schedule = self.unfreeze_schedule.get(
            self.experience_id, None)
        if unfreeze_schedule is None:
            return

        # Freeze modules at the beginning of training
        if epoch == 0:
            cold_modules = sum(unfreeze_schedule.values(), [])
            runner.logger.info(
                f"[LGFPlugin] Freezing modules: {cold_modules}")
            self.set_requires_grad(model, cold_modules, False)

        # Unfreeze modules at specified epochs
        warm_modules = unfreeze_schedule.get(epoch, [])
        if warm_modules:
            runner.logger.info(
                f"[LGFPlugin] Unfreezing modules at epoch {epoch}: {warm_modules}")
            self.set_requires_grad(model, warm_modules, True)

    def after_train_epoch(self, runner):
        """Log the status of components after training an epoch."""
        if len(self.frozen_modules) > 0:
            log_msg = f"[LGFPlugin] {len(self.frozen_modules)} modules frozen after epoch {runner.epoch}"
            runner.logger.info(log_msg)

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        super().before_backward(runner, experience_index, losses, data_batch)
        if self.scale_lwf:
            max_epochs = (
                self.max_epochs[experience_index]
                if isinstance(self.max_epochs, (list, tuple))
                else self.max_epochs
            )
            loss_scale = 1.0 + runner.epoch / max_epochs
            losses["loss_lwf"] *= loss_scale

    def after_experience(self, runner, experience_index: int):
        super().after_experience(runner, experience_index)
        next_experience = experience_index + 1
        if self.unfreeze_schedule.get(next_experience, None) == "auto":
            # Automatically generate unfreeze schedule for the next experience
            unfreeze_schedule = self.generate_unfreeze_schedule(runner, next_experience)
            self.unfreeze_schedule[next_experience] = unfreeze_schedule

    def generate_unfreeze_schedule(self, runner, experience_index: int) -> Dict[int, List[str]]:
        # Get max_epochs for the current experience
        max_epochs = (
            self.max_epochs[experience_index]
            if isinstance(self.max_epochs, (list, tuple))
            else self.max_epochs
        )

        # Get current previous model
        model = runner.module
        assert model is not None

        # Compute parameter importances
        runner.logger.info(
            f"[LGFPlugin] Computing layer-wise importances for E_{experience_index - 1}")
        importances = self.compute_importances(
            model,
            runner.train_loop.dataloader,
            runner.optim_wrapper,
            runner.device,
        )
        runner.logger.info(
            f"[LWFPlugin] Layer-wise importances for E_{experience_index - 1}: {importances}")

        # Compute unfreeze schedule
        unfreeze_schedule = dict()
        for name, importance in importances.items():
            # Determine the epoch at which the layer should be unfrozen
            unfreeze_epoch = int(max_epochs * importance)
            if unfreeze_epoch == 0:
                continue

            if name == "":
                continue

            # Add the layer to the unfreeze schedule
            if unfreeze_epoch not in unfreeze_schedule:
                unfreeze_schedule[unfreeze_epoch] = []
            unfreeze_schedule[unfreeze_epoch].append(name)

        runner.logger.info(
            f"[LGFPlugin] Unfreeze schedule for E_{experience_index}: {unfreeze_schedule}")
        return unfreeze_schedule

    def compute_importances(
        self,
        model,
        dataloader,
        optim_wrapper,
        device,
    ) -> Dict[str, Tuple[float, int]]:
        """
        Computes the EWC importance matrix for each layer of the model.

        The importance for each layer is computed by summing the squared gradients
        of all parameters within the layer and normalizing these values across layers.

        Args:
            model: The model to compute layer importances for.
            dataloader: DataLoader providing the dataset.
            optim_wrapper: The optimizer wrapper used for gradient computation.
            device: Device on which the model is running.
            max_epochs: The maximum number of epochs for which training will run.

        Returns:
            A dictionary mapping layer names to a tuple containing the normalized importance 
            and the suggested epoch number at which the layer should be unfrozen.
        """
        model.eval()

        # Handle RNN-like modules on GPU
        if device == torch.device("cuda"):
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.train()

        def ignore_name(name):
            """ Ignore model and top-level modules. """
            return name == "" or len(name.split(".")) == 1

        # Initialize importance storage
        layer_importances = {name: 0.0 for name,
                             _ in model.named_modules() if not ignore_name(name)}

        # Compute importances
        for data in tqdm(dataloader, desc="Computing layer importances"):
            optim_wrapper.optimizer.zero_grad()
            data = model.data_preprocessor(data, True)
            losses = model._run_forward(data, mode='loss')
            loss, _ = model.parse_losses(losses)
            loss.backward()

            # Sum squared gradients for each layer
            for name, module in model.named_modules():
                if ignore_name(name):
                    continue

                for param in module.parameters():
                    if param.grad is not None:
                        param_importance = param.grad.data.clone().pow(2).sum().item()
                        layer_importances[name] += param_importance

        # Remove zero-importance or ignored modules and average over mini-batch length
        length = float(len(dataloader))
        layer_importances = {name: importance / length for name, importance in layer_importances.items()
                             if not ignore_name(name) and importance > 0}

        # Use a sigmoid function to normalize importances
        def sigmoid(x): return 1 / (1 + torch.exp(-x))
        importances = torch.tensor(list(layer_importances.values()))
        for (name, _), importance in zip(model.named_modules(), sigmoid(importances)):
            layer_importances[name] = importance.item()

        return layer_importances
