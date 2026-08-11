import os
from typing import Optional

import mmcv
import numpy as np
import torch

from mmpose.registry import VISUALIZERS
from mmpose.visualization import PoseLocalVisualizer


@VISUALIZERS.register_module()
class DynamicPoseVisualizer(PoseLocalVisualizer):
    """Pose visualizer with dynamic keypoint radius and line width.

    Keypoint radius and skeleton line width are computed as a fraction of
    the image's larger dimension. Invisible keypoints (score < kpt_thr)
    are not drawn, but skeleton lines are always drawn regardless of
    endpoint visibility.
    """

    def __init__(self,
                 name='visualizer',
                 image=None,
                 vis_backends=None,
                 save_dir=None,
                 bbox_color='green',
                 kpt_color='red',
                 link_color=None,
                 text_color=(255, 255, 255),
                 skeleton=None,
                 line_width=1,
                 radius=3,
                 radius_scale=0.008,
                 line_width_scale=0.003,
                 show_keypoint_weight=False,
                 backend='opencv',
                 alpha=1.0,
                 save_by_category=False):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            kpt_color=kpt_color,
            link_color=link_color,
            text_color=text_color,
            skeleton=skeleton,
            line_width=line_width,
            radius=radius,
            show_keypoint_weight=show_keypoint_weight,
            backend=backend,
            alpha=alpha)
        self.radius_scale = radius_scale
        self.line_width_scale = line_width_scale
        self._base_radius = radius
        self._base_line_width = line_width
        self.save_by_category = save_by_category
        self.save_dir = save_dir

    def _compute_dynamic_sizes(self, image):
        """Compute dynamic radius and line width based on image size."""
        img_h, img_w = image.shape[:2]
        max_dim = max(img_h, img_w)
        dynamic_radius = max(1, int(max_dim * self.radius_scale))
        dynamic_line_width = max(1, int(max_dim * self.line_width_scale))
        return dynamic_radius, dynamic_line_width

    def _draw_instances_kpts(self,
                             image,
                             instances,
                             kpt_thr=0.3,
                             show_kpt_idx=False,
                             skeleton_style='mmpose'):
        """Draw keypoints and skeletons with dynamic sizing.

        Key differences from the base class:
        - radius and line_width are computed dynamically from image size.
        - Skeleton lines are drawn even if endpoint keypoints are invisible,
          but lines connecting to out-of-bounds points are still skipped.
        """
        if skeleton_style == 'openpose':
            return self._draw_instances_kpts_openpose(image, instances,
                                                      kpt_thr)

        self.set_image(image)
        img_h, img_w, _ = image.shape

        dynamic_radius, dynamic_line_width = self._compute_dynamic_sizes(image)

        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            for kpts, visible in zip(keypoints, keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw skeleton links (always draw, even if endpoints invisible)
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w
                                or pos1[1] <= 0 or pos1[1] >= img_h
                                or pos2[0] <= 0 or pos2[0] >= img_w
                                or pos2[1] <= 0 or pos2[1] >= img_h
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue

                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1, 0.5 *
                                    (visible[sk[0]] + visible[sk[1]])))
                        self.draw_lines(
                            X, Y, color, line_widths=dynamic_line_width)

                # draw each keypoint (skip invisible ones)
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([dynamic_radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=dynamic_radius)
                    if show_kpt_idx:
                        kpt_idx_coords = kpt + [
                            dynamic_radius, -dynamic_radius
                        ]
                        self.draw_texts(
                            str(kid),
                            kpt_idx_coords,
                            colors=color,
                            font_sizes=dynamic_radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

        return self.get_image()

    @staticmethod
    def _get_category_id(data_sample):
        """Extract category ID from a data sample."""
        cid = data_sample.get('category_id', None)
        if cid is not None:
            if isinstance(cid, (np.ndarray, torch.Tensor)):
                cid = cid.item()
            return int(cid)

        gt_instances = data_sample.get('gt_instances', None)
        if gt_instances is not None:
            if isinstance(gt_instances, dict):
                cid = gt_instances.get('category_id', None)
            else:
                cid = getattr(gt_instances, 'category_id', None)
            if cid is not None:
                if isinstance(cid, (np.ndarray, torch.Tensor)):
                    cid = cid.item()
                return int(cid)

        return -1

    def _get_category_ids(self, instances, data_sample=None):
        """Extract category IDs from instances object.

        Args:
            instances: gt_instances or pred_instances object.
            data_sample: PoseDataSample object containing metainfo with category_id.
        """
        category_ids = [-1]

        if instances is None:
            return category_ids

        cid = None
        if data_sample is not None:
            metainfo = getattr(data_sample, 'metainfo', {})
            if 'category_id' in metainfo:
                cid = metainfo['category_id']
                if isinstance(cid, (np.ndarray, torch.Tensor)):
                    cid = cid.item()
                if cid is not None:
                    cid = int(cid)

        if cid is None:
            if isinstance(instances, dict):
                cid = instances.get('category_id', None)
            else:
                cid = getattr(instances, 'category_id', None)

        if cid is not None:
            if isinstance(cid, (np.ndarray, torch.Tensor)):
                cid = cid.item()
            category_ids = [int(cid)] * len(instances) if hasattr(instances, '__len__') else [int(cid)]

        if isinstance(category_ids, (np.ndarray, torch.Tensor)):
            category_ids = category_ids.tolist()
        elif not isinstance(category_ids, list):
            category_ids = [category_ids]

        return category_ids

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample=None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_heatmap: bool = False,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save, optionally per-category."""
        if not self.save_by_category or data_sample is None:
            return super().add_datasample(
                name=name,
                image=image,
                data_sample=data_sample,
                draw_gt=draw_gt,
                draw_pred=draw_pred,
                draw_heatmap=draw_heatmap,
                draw_bbox=draw_bbox,
                show_kpt_idx=show_kpt_idx,
                skeleton_style=skeleton_style,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                kpt_thr=kpt_thr,
                step=step)

        gt_instances = data_sample.get('gt_instances', None)
        pred_instances = data_sample.get('pred_instances', None)

        gt_category_ids = self._get_category_ids(gt_instances, data_sample)
        pred_category_ids = self._get_category_ids(pred_instances, data_sample)

        all_category_ids = gt_category_ids + pred_category_ids
        unique_cats = sorted(set(c for c in all_category_ids if c >= 0))

        if not unique_cats:
            unique_cats = [-1]

        for cat_id in unique_cats:
            cat_img = image.copy()

            if gt_instances is not None and draw_gt and cat_id in gt_category_ids:
                cat_mask = np.array([c == cat_id for c in gt_category_ids])
                cat_instances = self._filter_instances(gt_instances, cat_mask)
                cat_img = super()._draw_instances_kpts(
                    cat_img, cat_instances, kpt_thr, show_kpt_idx, skeleton_style)
                if draw_bbox:
                    cat_img = super()._draw_instances_bbox(cat_img, cat_instances)

            if pred_instances is not None and draw_pred and cat_id in pred_category_ids:
                cat_mask = np.array([c == cat_id for c in pred_category_ids])
                cat_instances = self._filter_instances(pred_instances, cat_mask)
                cat_img = super()._draw_instances_kpts(
                    cat_img, cat_instances, kpt_thr, show_kpt_idx, skeleton_style)
                if draw_bbox:
                    cat_img = super()._draw_instances_bbox(cat_img, cat_instances)

            self._save_image(cat_img, name, cat_id, out_file, data_sample)

    @staticmethod
    def _get_category_name(cat_id, data_sample):
        """Get category name from category id.

        Args:
            cat_id: Category ID.
            data_sample: PoseDataSample object containing metainfo.

        Returns:
            str: Category name, or f'cat_{cat_id}' if not found.
        """
        if data_sample is not None:
            metainfo = getattr(data_sample, 'metainfo', {})
            categories = metainfo.get('categories', [])
            for cat in categories:
                cid = cat.get('id', None)
                if cid is not None:
                    if isinstance(cid, (np.ndarray, torch.Tensor)):
                        cid = cid.item()
                    if cid == cat_id:
                        return cat.get('name', f'cat_{cat_id}')
        return f'cat_{cat_id}'

    def _filter_instances(self, instances, mask):
        """Filter instances by boolean mask."""
        class InstanceWrapper:
            def __init__(self, data):
                self._data = data
            def get(self, key, default=None):
                return self._data.get(key, default) if isinstance(self._data, dict) else getattr(self._data, key, default)
            def __contains__(self, key):
                return key in self._data if isinstance(self._data, dict) else hasattr(self._data, key)
            def __getattr__(self, name):
                if name.startswith('_'):
                    return super().__getattribute__(name)
                if isinstance(self._data, dict):
                    return self._data.get(name)
                return getattr(self._data, name)

        if isinstance(instances, dict):
            filtered = {}
            for k, v in instances.items():
                if hasattr(v, '__len__') and len(v) == len(mask):
                    filtered[k] = v[mask]
                else:
                    filtered[k] = v
            return InstanceWrapper(filtered)
        else:
            filtered = {}
            for attr in dir(instances):
                if attr.startswith('_'):
                    continue
                try:
                    v = getattr(instances, attr)
                    if hasattr(v, '__len__') and len(v) == len(mask):
                        filtered[attr] = v[mask]
                except:
                    pass
            return InstanceWrapper(filtered)

    def _save_image(self, image, name, cat_id, out_file, data_sample=None):
        """Save image to file in category folder."""
        cat_name = self._get_category_name(cat_id, data_sample)

        if self.save_dir:
            base_dir = self.save_dir
        elif out_file:
            base_dir = os.path.dirname(out_file)
        else:
            base_dir = '.'

        if base_dir and base_dir != '.':
            os.makedirs(base_dir, exist_ok=True)
            cat_dir = os.path.join(base_dir, cat_name)
        else:
            cat_dir = cat_name
        os.makedirs(cat_dir, exist_ok=True)

        if out_file:
            base, ext = os.path.splitext(os.path.basename(out_file))
            save_path = os.path.join(cat_dir, f'{base}_{cat_name}{ext}')
        else:
            save_path = os.path.join(cat_dir, f'{name}_{cat_name}.jpg')

        mmcv.imwrite(image[..., ::-1], save_path)