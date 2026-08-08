import torch
import numpy as np
import cv2


@torch.no_grad()
def quad_aligned_bbox(quads: torch.Tensor, direction: torch.Tensor = None):
    """Compute oriented bounding boxes from quadrilateral keypoints.

    The direction vector is determined by the line from bottom_center (points 1,2)
    to top_center (points 0,3). When `direction` is provided, it overrides the
    internally computed direction (useful for ensuring GT and pred use the same orientation).

    Args:
        quads: Quadrilateral keypoints, shape [B, 4, 2] (counter-clockwise: 0=左上, 1=左下, 2=右下, 3=右上).
        direction: Optional external direction vector [B, 2], overriding the internal computation.

    Returns:
        W: Width of oriented bbox [B].
        H: Height of oriented bbox [B].
        bbox_corners: 4 corners of oriented bbox [B, 4, 2].
        centers: Center of oriented bbox [B, 2].
    """
    B = quads.shape[0]
    eps = 1e-8

    if direction is not None:
        directions = direction
    else:
        # 四点逆时针排列: 0=左上, 1=左下, 2=右下, 3=右上
        # 方向向量：上面两点中心 - 下面两点中心
        top_center = (quads[:, 0] + quads[:, 3]) / 2    # 左上 + 右上
        bottom_center = (quads[:, 1] + quads[:, 2]) / 2  # 左下 + 右下
        directions = top_center - bottom_center

    # 归一化方向向量
    vec_norm = torch.norm(directions, dim=-1, keepdim=True)
    u = directions / (vec_norm + eps)
    v = torch.stack([-u[..., 1], u[..., 0]], dim=-1)

    # 投影 s//u, t//v
    s = torch.einsum("bki,bi->bk", quads, u)
    t = torch.einsum("bki,bi->bk", quads, v)

    s_min, s_max = s.min(dim=-1)[0], s.max(dim=-1)[0]
    t_min, t_max = t.min(dim=-1)[0], t.max(dim=-1)[0]

    W = s_max - s_min
    H = t_max - t_min

    # 生成4个角点 st
    st_corners = torch.stack([
        torch.stack([s_min, t_min], -1),
        torch.stack([s_max, t_min], -1),
        torch.stack([s_max, t_max], -1),
        torch.stack([s_min, t_max], -1),
    ], dim=1)  # [B,4,2]

    s = st_corners[..., 0:1]
    t = st_corners[..., 1:2]
    x = s * u[:, None, 0:1] + t * v[:, None, 0:1]
    y = s * u[:, None, 1:2] + t * v[:, None, 1:2]
    bbox_corners = torch.cat([x, y], dim=-1)
    # 中心
    sc = (s_min + s_max) / 2
    tc = (t_min + t_max) / 2
    cx = sc * u[:, 0] + tc * v[:, 0]
    cy = sc * u[:, 1] + tc * v[:, 1]
    centers = torch.stack([cx, cy], -1)
    return W, H, bbox_corners, centers


def quad_center_dist(gt_quads: torch.Tensor, pred_quads: torch.Tensor):
    """批量四边形质心欧氏距离 [B,]"""
    c_gt = gt_quads.mean(dim=1)   # [B,2]
    c_pred = pred_quads.mean(dim=1)
    dist = torch.norm(c_gt - c_pred, dim=-1)
    return dist


def quad_center(quad: np.ndarray):
    """四边形质心（四点平均）"""
    return quad.mean(axis=0)
