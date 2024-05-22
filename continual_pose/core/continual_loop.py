import os
import os.path as osp
import shutil
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.loops import EpochBasedTrainLoop, ValLoop, TestLoop


@LOOPS.register_module()
class CLEpochBasedTrainLoop(EpochBasedTrainLoop):
    """Loop for epoch-based training in continual learning.

    This extends the EpochBasedTrainLoop with an additional 'before_backward'
    hook for modifying the loss function before the backward pass. This is
    useful for strategies that require modifying the loss function, such as
    Elastic Weight Consolidation (EWC) and Synaptic Intelligence (SI).

    NOTE: Models that require a custom train_step are not supported. Only the
    default implementation from mmengine.models.BaseModel is used.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)

    def _train_step(self, runner, idx, data: Union[dict, tuple, list]) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``runner.model.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``runner.model(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``runner.model.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``runner.optim_wrapper.update_params(loss)`` to update model.

        The only difference between this method and the :meth:`train_step` in
        :class:`BaseModel` is the addition of a 'before_backward' hook that
        is called with the ``parsed_losses`` tensor before the backward pass.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        model = runner.module

        # Enable automatic mixed precision training context.
        with runner.optim_wrapper.optim_context(self):
            data = model.data_preprocessor(data, True)
            losses = model._run_forward(data, mode='loss')  # type: ignore

        # Call before_backward hook
        # This hook is used by regularization-based plugins to add
        # a penalty to the loss for mitigating forgetting
        runner.call_hook('before_backward',
                         experience_index=runner.train_loop.experience_id,
                         losses=losses,
                         data_batch=data)

        parsed_losses, log_vars = model.parse_losses(losses)  # type: ignore
        loss = runner.optim_wrapper.scale_loss(parsed_losses)
        runner.optim_wrapper.backward(loss)
        self.runner.call_hook(
            'after_backward',
            batch_idx=idx,
            data_batch=data)

        if runner.optim_wrapper.should_update():
            runner.optim_wrapper.step()
            runner.optim_wrapper.zero_grad()

        return log_vars

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter',
            batch_idx=idx,
            data_batch=data_batch)

        outputs = self._train_step(self.runner, idx, data_batch)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


@LOOPS.register_module()
class ContinualTrainingLoop(BaseLoop):
    """Loop for continual learning, managing multiple datasets sequentially.

    Args:
        runner (Runner): A reference of runner.
        dataloaders (List[Union[DataLoader, Dict]]): A list of dataloaders or dictionaries
            to build dataloaders, one for each learning experience.
        max_epochs_per_experience (Union[int, List, Tuple]): Total training epochs for each experience.
            If a list is provided, it should have the same length as `dataloaders` and specify
            the number of epochs for each experience. If an integer is provided, all experiences
            will have the same number of epochs.
    """

    def __init__(
            self,
            runner,
            num_experiences: int,
            max_epochs_per_experience: Union[int, List, Tuple],
            val_interval: int = 1) -> None:
        super().__init__(runner, None)  # No single dataloader applies to all experiences
        self.num_experiences = num_experiences
        if isinstance(max_epochs_per_experience, int):
            max_epochs_per_experience = [
                max_epochs_per_experience] * num_experiences
        self.max_epochs_per_experience = max_epochs_per_experience
        self.val_interval = val_interval

        self.experience_id = -1
        self.experience = None

    def get_info(self, index):
        return dict(
            id=index,
            name=f'Experience {index + 1}',
            max_epochs=self.max_epochs_per_experience[index],
            val_interval=self.val_interval,
        )

    def _build_dataloader(self, dataloader: Union[DataLoader, Dict]) -> DataLoader:
        if isinstance(dataloader, dict):
            return self.runner.build_dataloader(dataloader)
        return dataloader

    def build_experience_train_loop(self, dataloader, max_epochs, val_interval) -> CLEpochBasedTrainLoop:
        return CLEpochBasedTrainLoop(
            runner=self.runner,
            dataloader=dataloader,
            max_epochs=max_epochs,
            val_interval=val_interval,
        )

    def build_experience_val_loop(self, dataloader, evaluator) -> ValLoop:
        return ValLoop(
            runner=self.runner,
            dataloader=dataloader,
            evaluator=evaluator,
        )

    def build_experience_test_loop(self, dataloader, evaluator) -> TestLoop:
        return TestLoop(
            runner=self.runner,
            dataloader=dataloader,
            evaluator=evaluator,
        )

    def run_experience(self, index):
        info = self.get_info(index)
        info.update(self.runner.get_experience_data(index))

        if 'train_dataloader' in info:
            train_loop = self.build_experience_train_loop(
                self._build_dataloader(info['train_dataloader']),
                info['max_epochs'],
                info['val_interval'],
            )
        if 'val_dataloader' in info:
            self.runner._val_loop = self.build_experience_val_loop(
                self._build_dataloader(info['val_dataloader']),
                info['val_evaluator'],
            )
        if 'test_dataloader' in info:
            self.runner._test_loop = self.build_experience_test_loop(
                self._build_dataloader(info['test_dataloader']),
                info['test_evaluator'],
            )

        self.experience_id = index
        self.experience = train_loop
        self.dataloader = train_loop.dataloader

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.runner.optim_wrapper = self.runner.build_optim_wrapper(
            self.runner._optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.runner.scale_lr(self.runner.optim_wrapper,
                             self.runner._auto_scale_lr)

        if self.runner._param_schedulers is not None:
            self.runner.param_schedulers = self.runner.build_param_scheduler(  # type: ignore
                self.runner._param_schedulers)  # type: ignore

        # Initiate inner count of `optim_wrapper`.
        self.runner.optim_wrapper.initialize_count_status(
            self.runner.model,
            train_loop.iter,  # type: ignore
            train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self.runner._maybe_compile('train_step')

        self.runner.call_hook('before_experience', experience_index=index)
        model = train_loop.run()
        self.runner.call_hook('after_experience', experience_index=index)

        return model

    def run(self) -> None:
        """Execute training across all experiences."""
        runner = self.runner

        self.experience_id = -1
        runner.call_hook('before_first_experience')
        base_work_dir = osp.join(runner.work_dir, 'experiences')
        for index in range(self.num_experiences):
            # Create experience-specific work_dir
            experience_dir = osp.join(base_work_dir, f'{index}')
            os.makedirs(experience_dir, exist_ok=True)

            # Run the current experience
            self.run_experience(index)

            # Move all checkpoints to experience directory
            ckpt_files = [f for f in os.listdir(runner.work_dir)
                          if f.endswith('.pth') and f.startswith('epoch')]
            for ckpt_file in ckpt_files:
                src = osp.join(runner.work_dir, ckpt_file)
                dst = osp.join(experience_dir, ckpt_file)
                shutil.move(src, dst)

    @property
    def max_epochs(self) -> int:
        """Total training epochs for the current experience."""
        if self.experience is None:
            return self.max_epochs_per_experience[0]
        return self.experience.max_epochs

    @property
    def max_iters(self) -> int:
        """Total training iterations for the current experience."""
        if self.experience is None:
            return self.max_epochs
        return self.experience.max_iters

    @property
    def epoch(self) -> int:
        """Current epoch for the current experience."""
        if self.experience is None:
            return 0
        return self.experience.epoch

    @property
    def iter(self) -> int:
        """Current iteration for the current experience."""
        if self.experience is None:
            return 0
        return self.experience.iter
