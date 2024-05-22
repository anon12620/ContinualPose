from typing import Dict

import torch
from mmpose.registry import HOOKS, TRANSFORMS

from .base import BasePlugin


@HOOKS.register_module()
class LWFPlugin(BasePlugin):
    """Learning without Forgetting (LwF) Plugin.

    LFL uses a distillation loss between logits of the current and previous
    model to mitigate catastrophic forgetting.

    This plugin does not use task identities.

    Args:
        temperature (float): Temperature for distillation loss.
        lambda_lwf (float or list[float]): Weight for distillation loss.
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

    See Also:
        - Learning without Forgetting
            https://arxiv.org/abs/1606.09282
    """

    def __init__(
            self,
            temperature: float = 2.0,
            lambda_lwf: float = 0.5,
            converters: Dict[int, Dict] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_lwf = lambda_lwf
        self.prev_model = None

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

    def compute_distillation_loss(self, logits_curr, logits_prev, temperature):
        """
        Compute distillation loss between current and previous model logits.
        """
        # Softmax of logits
        probs_curr = self.softmax(logits_curr / temperature)
        probs_prev = self.softmax(logits_prev / temperature)

        # Compute KL divergence
        loss = self.kldiv(probs_curr.log(), probs_prev)
        return loss

    def before_experience(self, runner, experience_index: int):
        super().before_experience(runner, experience_index)
        self.prev_model = runner.last_model
        if experience_index > 0:
            assert self.prev_model is not None, (
                "The previous model is required for LwF. Please include a "
                "SnapshotPlugin in your config to save the previous model."
            )

            # Set to training mode
            self.prev_model.train()

    def after_experience(self, runner, experience_index: int):
        super().after_experience(runner, experience_index)
        if self.prev_model is not None:
            self.prev_model.eval()  # Restore to eval mode

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        """
        Add euclidean loss between prev and current features as penalty
        """
        lambda_lwf = (
            self.lambda_lwf[experience_index]
            if isinstance(self.lambda_lwf, (list, tuple))
            else self.lambda_lwf
        )

        if self.prev_model is not None:
            current_model = runner.module
            prev_model = self.prev_model

            # Get student predictions
            # NOTE: This results in an additional forward pass through the head,
            #       which is not ideal. However, MMPose does not currently support
            #       returning the predictions from the forward pass in the 'loss'
            #       mode, so we have to do this for a model-agnostic implementation
            #       of LwF.
            feats_curr = current_model.extract_feat(data_batch['inputs'])
            preds_curr = current_model.head.forward(feats_curr)

            # Convert student predictions to teacher format if necessary
            if experience_index in self.converters:
                c = self.converters[experience_index]
                def _convert(preds):
                    preds[:, c.target_index, :] = preds[:, c.source_index, :]
                    preds = preds[:, :c.num_keypoints, :]
                    return preds

                if isinstance(preds_curr, (tuple, list)):
                    preds_curr = [_convert(p) for p in preds_curr]
                else:
                    preds_curr = _convert(preds_curr)

            # Get teacher predictions
            with torch.no_grad():
                feats_last = prev_model.extract_feat(data_batch['inputs'])
                preds_last = prev_model.head.forward(feats_last)

            # Compute LwF penalty
            if isinstance(preds_curr, (tuple, list)):
                penalty = torch.tensor(0.0, device=runner.device)
                for p_curr, p_last in zip(preds_curr, preds_last):
                    penalty += self.compute_distillation_loss(
                        p_curr, p_last, self.temperature)
            else:
                penalty = self.compute_distillation_loss(
                    preds_curr, preds_last, self.temperature)

            # Add penalty to losses dictionary
            losses["loss_kpt"] = losses["loss_kpt"] * (1 - lambda_lwf)
            losses["loss_lwf"] = penalty * lambda_lwf
