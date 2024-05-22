from collections import defaultdict
from typing import Dict, Tuple, Union

import torch
from mmpose.registry import HOOKS, TRANSFORMS
from tqdm import tqdm

# Assuming LayerEWCPlugin is in ewc.py
from .base import BasePlugin
from ..utils import ParamData


@HOOKS.register_module()
class IWDPlugin(BasePlugin):
    """ Importance Weighted Distillation (IWD) Plugin.

    Args:
        lambda_ewc (float): Weight for EWC penalty.
        temperature (float): Base temperature for distillation loss.
        lambda_iwd (float): Weight for distillation loss.
        decay_factor (float): A decay multiplier applied to the importance values,
                              reducing the influence of older tasks' importance over time.
        converters (Dict[Dict]): Dictionary with experience index as key and
                                 'KeypointConverter' configuration as value.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        lambda_iwd: float = 0.5,
        converters: Dict[int, Dict] = None,
        scale_layer_temp: bool = True,
        scale_by_heuristic: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_iwd = lambda_iwd
        self.prev_model = None
        self.scale_layer_temp = scale_layer_temp
        self.scale_by_heuristic = scale_by_heuristic

        # Distillation loss
        self.softmax = torch.nn.Softmax(dim=1)
        self.kldiv = torch.nn.KLDivLoss(reduction="batchmean")

        # Initialize converters
        self.converters = dict()
        if converters is not None:
            for i, c in converters.items():
                self.converters[i] = TRANSFORMS.build(dict(
                    type="KeypointConverter",
                    num_keypoints=c["num_keypoints"],
                    mapping=c["mapping"],
                ))

        self.layer_importances = defaultdict(float)

        self.hooks = []
        self.layer_outputs_current = {}
        self.layer_outputs_prev = {}

    def compute_distillation_loss(self, logits_curr, logits_prev, temperature):
        """
        Compute distillation loss between current and previous model logits with layer-wise temperature.
        """
        # Softmax of logits
        probs_curr = self.softmax(logits_curr / temperature)
        probs_prev = self.softmax(logits_prev / temperature)

        # Compute KL divergence
        loss = self.kldiv(probs_curr.log(), probs_prev)
        return loss

    def register_current_hooks(self, model, logger=None):
        def get_hook(layer_name, output_dict, logger=None):
            def hook(module, input, output):
                output_dict[layer_name] = output
                return output
            return hook

        for name, module in model.named_modules():
            logger.info(f"[IWD2Plugin] Registering current hook for {name}")
            self.layer_outputs_current[name] = None
            hook = get_hook(name, self.layer_outputs_current, logger)
            hook = module.register_forward_hook(hook)
            self.hooks.append(hook)

    def register_prev_hooks(self, model, logger=None):
        def get_hook(layer_name, output_dict, logger=None):
            def hook(module, input, output):
                output_dict[layer_name] = output
                return output
            return hook

        for name, module in model.named_modules():
            logger.info(f"[IWD2Plugin] Registering previous hook for {name}")
            self.layer_outputs_prev[name] = None
            hook = get_hook(name, self.layer_outputs_prev, logger)
            hook = module.register_forward_hook(hook)
            self.hooks.append(hook)
        logger.info(f"[IWD2Plugin] Registered hooks: {len(self.hooks)}")

    def remove_hooks(self):
        """
        Remove all registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def before_experience(self, runner, experience_index: int):
        super().before_experience(runner, experience_index)
        self.prev_model = runner.last_model
        if experience_index > 0:
            assert self.prev_model is not None, (
                "The previous model is required for IWD. Please include a "
                "SnapshotPlugin in your config to save the previous model."
            )

            # Set to training mode
            self.prev_model.train()

            self.register_prev_hooks(self.prev_model, runner.logger)
            runner.logger.info(f"[IWD2Plugin] Registered M_{experience_index-1} hooks: {self.layer_outputs_prev.keys()}")

            # Register hooks for current model
            current_model = runner.module
            self.register_current_hooks(current_model, runner.logger)
            runner.logger.info(f"[IWD2Plugin] Registered M_{experience_index} hooks: {self.layer_outputs_current.keys()}")

    def after_experience(self, runner, experience_index):
        super().after_experience(runner, experience_index)
        if self.prev_model is not None:
            self.prev_model.eval()  # Restore to eval mode

        runner.logger.info(f"[IWD2Plugin] Removing hooks and clearing model outputs after E_{experience_index}")
        self.remove_hooks()
        self.layer_outputs_current = {}
        self.layer_outputs_prev = {}

        runner.logger.info(f"[IWD2Plugin] Computing layer importances after E_{experience_index}")
        layer_importances = self.compute_importances(
            runner.module,
            runner.train_loop.dataloader,
            runner.optim_wrapper,
            runner.device,
        )
        self.layer_importances.update(layer_importances)
        runner.logger.info(f"[IWD2Plugin] Layer importances: {layer_importances}")
    
    def before_backward(self, runner, experience_index, losses, data_batch=None):
        """
        Add layer-wise distillation loss between prev and current features as penalty.
        """
        lambda_iwd = self.lambda_iwd

        if self.prev_model is not None:
            current_model = runner.module
            prev_model = self.prev_model

            # Extract current model features and predictions
            feats_curr = current_model.extract_feat(data_batch['inputs'])
            preds_curr = current_model.head.forward(feats_curr)

            # Convert student predictions to teacher format if necessary
            c = self.converters.get(experience_index, None)
            def _convert(preds):
                if c is None:
                    # No conversion needed
                    return preds

                # Avoid in-place modification
                preds = preds.clone()
                preds[:, c.target_index, :] = preds[:, c.source_index, :]
                preds = preds[:, :c.num_keypoints, :]
                return preds

            if isinstance(preds_curr, (tuple, list)):
                preds_curr = [_convert(p) for p in preds_curr]
            else:
                preds_curr = _convert(preds_curr)

            # Extract previous model features and predictions
            with torch.no_grad():
                feats_last = prev_model.extract_feat(data_batch['inputs'])
                preds_last = prev_model.head.forward(feats_last)

            # Compute distillation loss
            if isinstance(preds_curr, (tuple, list)):
                penalty = torch.tensor(0.0, device=runner.device)
                for p_curr, p_last in zip(preds_curr, preds_last):
                    penalty += self.compute_distillation_loss(
                        p_curr, p_last, self.temperature)
            else:
                penalty = self.compute_distillation_loss(
                    preds_curr, preds_last, self.temperature)

            # Compute layer-wise distillation loss
            penalty_lw = None
            if experience_index > 0:
                penalty_lw = torch.tensor(0.0, device=runner.device)
                for layer_name, importance in self.layer_importances.items():
                    try:
                        if self.scale_by_heuristic:
                            epoch_importance = 0.05 + runner.epoch / 1000  # Ranges from 0.05 to 0.1
                            layer_temp = self.temperature * 1. / epoch_importance
                        elif self.scale_layer_temp:
                            layer_temp = self.temperature * 1. / importance
                        else:
                            layer_temp = self.temperature * 1. / 0.1
                        current_model_outputs = self.layer_outputs_current[layer_name]
                        previous_model_outputs = self.layer_outputs_prev[layer_name]

                        # convert current model outputs to previous model format
                        if current_model_outputs.shape[1] != previous_model_outputs.shape[1]:
                            current_model_outputs = _convert(current_model_outputs)

                        penalty_lw += self.compute_distillation_loss(
                            current_model_outputs,
                            previous_model_outputs,
                            layer_temp,
                        )
                    except Exception:
                        pass

            # Add penalty to losses dictionary
            losses["loss_kpt"] = losses["loss_kpt"] * (1 - lambda_iwd)
            losses["loss_lwf"] = penalty * lambda_iwd
            if penalty_lw is not None:
                losses["loss_iwd"] = penalty_lw

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
        for i, data in tqdm(enumerate(dataloader), desc="Computing layer importances"):
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
        normalized_importances = {}
        for name, importance in layer_importances.items():
            if ignore_name(name) or importance == 0:
                continue

            normalized_importances[name] = importance / length

        return normalized_importances
