import copy
from typing import Dict, Union

from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
from mmengine.model import is_model_wrapper
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner import Runner

from .continual_loop import ContinualTrainingLoop


ConfigType = Union[Dict, Config, ConfigDict]


class ContinualLearningRunner(Runner):
    def __init__(self,
                 *args,
                 train_dataloaders=None,
                 val_dataloaders=None,
                 test_dataloaders=None,
                 val_evaluators=None,
                 test_evaluators=None,
                 **kwargs):
        if train_dataloaders is not None:
            kwargs['train_dataloader'] = train_dataloaders[0]
        if val_dataloaders is not None:
            kwargs['val_dataloader'] = val_dataloaders[0]
        if test_dataloaders is not None:
            kwargs['test_dataloader'] = test_dataloaders[0]
        if val_evaluators is not None:
            kwargs['val_evaluator'] = val_evaluators[0]
        if test_evaluators is not None:
            kwargs['test_evaluator'] = test_evaluators[0]

        super().__init__(*args, **kwargs)
        self._train_dataloaders = train_dataloaders

        self._val_dataloaders = val_dataloaders
        self._val_evaluators = val_evaluators

        self._test_dataloaders = test_dataloaders
        self._test_evaluators = test_evaluators

        self.memory = None

        # Lazy initialization of `optim_wrapper` and `param_scheduler`
        self._optim_wrapper = kwargs.get('optim_wrapper')
        self._auto_scale_lr = kwargs.get('auto_scale_lr')

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        param_scheduler = kwargs.get('param_scheduler')
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self._param_schedulers = param_scheduler

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloaders=cfg.get('train_dataloaders'),
            val_dataloaders=cfg.get('val_dataloaders'),
            test_dataloaders=cfg.get('test_dataloaders'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluators=cfg.get('val_evaluators'),
            test_evaluators=cfg.get('test_evaluators'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner

    @property
    def last_model(self):
        if self.memory is None or not isinstance(self.memory, dict):
            return self.module

        last_model = self.memory.get('model', self.model)
        if isinstance(last_model, list) and len(last_model) > 0:
            return last_model[-1]

        return last_model

    @property
    def last_state_dict(self):
        if self.memory is None or not isinstance(self.memory, dict):
            return None

        state_dict = self.memory.get('state_dict', None)
        if isinstance(state_dict, list) and len(state_dict) > 0:
            return state_dict[-1]

        return state_dict

    @property
    def module(self):
        if is_model_wrapper(self.model):
            return self.model.module
        else:
            return self.model

    @property
    def device(self):
        if is_model_wrapper(self.model):
            return self.model.device
        else:
            get_device()

    def get_experience_data(self, index: int) -> Dict:
        info = dict()
        if self._train_dataloaders:
            info['train_dataloader'] = self._train_dataloaders[index]
        if self._val_dataloaders:
            info['val_dataloader'] = self._val_dataloaders[index]
            info['val_evaluator'] = self._val_evaluators[index]
        if self._train_dataloaders:
            info['test_dataloader'] = self._test_dataloaders[index]
            info['test_evaluator'] = self._test_evaluators[index]
        return info

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        if isinstance(loop, dict):
            loop_cfg = copy.deepcopy(loop)

            # Special handling for ContinualTrainingLoop
            if 'type' in loop_cfg and loop_cfg['type'] == 'ContinualTrainingLoop':
                # Build each dataloader specified for the different experiences
                return ContinualTrainingLoop(
                    runner=self,
                    num_experiences=len(self._train_dataloaders),
                    max_epochs_per_experience=loop_cfg.get(
                        'max_epochs_per_experience', 1),
                    val_interval=loop_cfg.get('val_interval', 1)
                )

        # Fallback to the default implementation for other loop types
        return super().build_train_loop(loop)

    def train(self):
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()

        # try to enable activation_checkpointing feature
        modules = self.cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return self.model

    def switch_model(self, model):
        self.model = self.build_model(model)
        self.model = self.wrap_model(self.cfg.get('model_wrapper_cfg'),
                                     self.model)

        # initialize the model weights
        self._init_model_weights()

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
