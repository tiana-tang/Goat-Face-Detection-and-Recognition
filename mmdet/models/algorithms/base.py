# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv.fileio
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule

from ..builder import (ALGORITHMS,build_detector)
import copy
# from ..builder impoer 

@ALGORITHMS.register_module()
class BaseAlgorithm(BaseModule):
    """Base class for algorithms, it consists of two main parts: architecture
    and algorithm components.

    Args:
        architecture (dict): Config for architecture to be slimmed.
        mutator (dict): Config for mutator, which is an algorithm component
            for NAS.
        pruner (dict): Config for pruner, which is an algorithm component
            for pruning.
        baseline (dict): Config for pruner, which is an algorithm component
            for knowledge distillation.
        retraining (bool): Whether is in retraining stage, if False,
            it is in pre-training stage.
        init_cfg (dict): Init config for ``BaseModule``.
        mutable_cfg (dict): Config for mutable of the subnet searched out,
            it will be needed in retraining stage.
        channel_cfg (dict): Config for channel of the subnet searched out,
            it will be needed in retraining stage.
    """

    def __init__(
        self,
        cfg=None,
        model_cfg=None,
        # train_cfg=None,
        # test_cfg=None,
        init_cfg=None,
    ):
        super(BaseAlgorithm, self).__init__(init_cfg)
        self.tuning = build_detector(model_cfg)
        
        self._init_baseline()

   
    def _init_baseline(self):
        self.baseline = copy.deepcopy(self.tuning)

    @property
    def with_baseline(self):
        """Whether or not this property exists."""
        return hasattr(self, 'baseline') and self.baseline is not None

    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        pass

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        return self.architecture.simple_test(img, img_metas)

    def show_result(self, img, result, **kwargs):
        """Draw `result` over `img`"""
        return self.architecture.show_result(img, result, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def tuning_step(self, data, optimizer):
        # with mp.get_context('spawn').Pool() as pool:
        #     res_baseline = pool.map(model_baseline.tuning_val_step, data)
            # res_baseline = ()
        # with torch.no_grad(): 
        # res_baseline = self.baseline.tuning_val_step(data) #输出baseline的detection结果
        res_baseline =None
        # res_tuning = self.tuning_val_step(data) #输出baseline的detection结果
        losses = self.tuning(**data, res_baseline=res_baseline)
        loss, log_vars = self.tuning._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs