import torch
from mmpose.registry import HOOKS

from .base import BasePlugin


@HOOKS.register_module()
class LFLPlugin(BasePlugin):
    """Less-Forgetful Learning (LFL) Plugin.

    LFL satisfies two properties to mitigate catastrophic forgetting.
        1) To keep the decision boundaries unchanged
        2) The feature space should not change much on target (new) data

    LFL uses euclidean loss between features from current and previous version
    of model as regularization to maintain the feature space and avoid
    catastrophic forgetting.

    This plugin does not use task identities.

    Args:
        alpha (float or list[float]): Euclidean loss hyper parameter.
            If a list is provided, it should have the same length as the number
            of experiences. If a single value is provided, it will be used for
            all experiences.

    See Also:
        - Less-forgetting Learning in Deep Neural Networks
          https://arxiv.org/abs/1607.00122
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.prev_model = None

    def before_experience(self, runner, experience_index: int):
        self.prev_model = runner.last_model

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        """
        Add euclidean loss between prev and current features as penalty
        """
        alpha = (
            self.alpha[experience_index]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )

        if self.prev_model is not None:
            # Set models in eval mode
            current_model = runner.module
            prev_model = self.prev_model
            current_model.eval()
            prev_model.eval()

            # Extract features (using the last layer features only)
            feats_curr = current_model.extract_feat(data_batch['inputs'])[-1]  
            feats_prev = prev_model.extract_feat(data_batch['inputs'])[-1]

            # Restore models to train mode
            current_model.train()

            # Compute LfL penalty (euclidean distance between features)
            penalty = torch.nn.functional.mse_loss(feats_curr, feats_prev)

            # Add penalty to losses dictionary
            losses["loss_kpt"] = losses["loss_kpt"] * (1 - alpha)
            losses["loss_lfl"] = penalty * alpha
