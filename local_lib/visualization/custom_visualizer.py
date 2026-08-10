import numpy as np
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
                 alpha=1.0):
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