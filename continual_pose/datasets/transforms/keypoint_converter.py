# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class KeypointConverter2(BaseTransform):
    """Change the order of keypoints according to the given mapping.

    Required Keys:

        - keypoints
        - keypoints_visible

    Modified Keys:

        - keypoints
        - keypoints_visible

    Args:
        num_keypoints (int): The number of keypoints in target dataset.
        mapping (list): A list containing mapping indexes. Each element has
            format (source_index, target_index)

    Example:
        >>> import numpy as np
        >>> # case 1: 1-to-1 mapping
        >>> # (0, 0) means target[0] = source[0]
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         (0, 0), (1, 1), (2, 2), (3, 3)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
        >>> assert np.equal(results['keypoints'],
        >>>                 np.arange(34).reshape(2, 3, 2)).all()
        >>> assert np.equal(results['keypoints_visible'],
        >>>                 np.arange(34).reshape(2, 3, 2) % 2).all()
        >>>
        >>> # case 2: 2-to-1 mapping
        >>> # ((1, 2), 0) means target[0] = (source[1] + source[2]) / 2
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         ((1, 2), 0), (1, 1), (2, 2)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
    """

    def __init__(self, num_keypoints: int,
                 mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                  int]]]):
        self.num_keypoints = num_keypoints
        self.mapping = mapping
        if len(mapping):
            source_index, target_index = zip(*mapping)
        else:
            source_index, target_index = [], []

        src1, src2 = [], []
        interpolation = False
        for x in source_index:
            if isinstance(x, (list, tuple)):
                assert len(x) == 2, 'source_index should be a list/tuple of ' \
                                    'length 2'
                src1.append(x[0])
                src2.append(x[1])
                interpolation = True
            else:
                src1.append(x)
                src2.append(x)

        # When paired source_indexes are input,
        # keep a self.source_index2 for interpolation
        if interpolation:
            self.source_index2 = src2

        self.source_index = src1
        self.target_index = list(target_index)
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transforms the keypoint results to match the target keypoints."""
        if isinstance(results, dict):
            num_instances = results['keypoints'].shape[0]
            if len(results['keypoints_visible'].shape) > 2:
                results['keypoints_visible'] = results['keypoints_visible'][:, :, 0]

            # Initialize output arrays
            keypoints = np.zeros((num_instances, self.num_keypoints, 3))
            keypoint_scores = np.zeros((num_instances, self.num_keypoints))
            keypoints_visible = np.zeros((num_instances, self.num_keypoints))
            key = 'keypoints_3d' if 'keypoints_3d' in results else 'keypoints'
            c = results[key].shape[-1]

            flip_indices = results.get('flip_indices', None)

            # Create a mask to weight visibility loss
            keypoints_visible_weights = keypoints_visible.copy()
            keypoints_visible_weights[:, self.target_index] = 1.0

            # Interpolate keypoints if pairs of source indexes provided
            if self.interpolation:
                keypoints[:, self.target_index, :c] = 0.5 * (
                    results[key][:, self.source_index] +
                    results[key][:, self.source_index2])

                if 'keypoint_scores' in results:
                    keypoint_scores[:, self.target_index] = 0.5 * (
                        results['keypoint_scores'][:, self.source_index] +
                        results['keypoint_scores'][:, self.source_index2])

                keypoints_visible[:, self.target_index] = results[
                    'keypoints_visible'][:, self.source_index] * results[
                        'keypoints_visible'][:, self.source_index2]
                # Flip keypoints if flip_indices provided
                if flip_indices is not None:
                    for i, (x1, x2) in enumerate(
                            zip(self.source_index, self.source_index2)):
                        idx = flip_indices[x1] if x1 == x2 else i
                        flip_indices[i] = idx if idx < self.num_keypoints else i
                    flip_indices = flip_indices[:len(self.source_index)]
            # Otherwise just copy from the source index
            else:
                keypoints[:,
                        self.target_index, :c] = results[key][:,
                                                                self.source_index]
                if 'keypoint_scores' in results:
                    keypoint_scores[:, self.target_index] = results[
                        'keypoint_scores'][:, self.source_index]
                keypoints_visible[:, self.target_index] = results[
                    'keypoints_visible'][:, self.source_index]

            # Update the results dict
            results['keypoints'] = keypoints[..., :2]
            if 'keypoint_scores' in results:
                results['keypoint_scores'] = keypoint_scores
            results['keypoints_visible'] = np.stack(
                [keypoints_visible, keypoints_visible_weights], axis=2)
            if 'keypoints_3d' in results:
                results['keypoints_3d'] = keypoints
                # results['lifting_target'] = keypoints[results['target_idx']]
                # results['lifting_target_visible'] = keypoints_visible[results['target_idx']]
            if flip_indices is not None:
                results['flip_indices'] = flip_indices

            return results
        elif isinstance(results, torch.Tensor):
            num_instances = results.shape[0]

            # Initialize output arrays
            keypoints = torch.zeros(
                (num_instances, self.num_keypoints, *results.shape[2:]), 
                device=results.device, requires_grad=results.requires_grad)

            # Interpolate keypoints if pairs of source indexes provided
            if self.interpolation:
                keypoints[:, self.target_index] = 0.5 * (
                    results[:, self.source_index] +
                    results[:, self.source_index2])
            # Otherwise just copy from the source index
            else:
                keypoints[:, self.target_index] = results[:, self.source_index]

            return keypoints
        else:
            raise ValueError('Invalid type of results', type(results))


    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(num_keypoints={self.num_keypoints}, '\
                    f'mapping={self.mapping})'
        return repr_str