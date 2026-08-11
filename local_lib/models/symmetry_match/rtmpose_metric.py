import numpy as np
import torch

from mmengine.logging import MMLogger
from pycocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.evaluation.metrics import CocoMetric
from mmpose.registry import METRICS
from local_lib.data.mix_dataset.coco_merge import merge_coco_categories


@METRICS.register_module()
class SymmetryMatchCocoMetric(CocoMetric):
    """COCO metric with category-aware symmetry keypoint matching.

    For symmetric keypoint pairs (e.g., lefttop/righttop), the predicted
    keypoints are swapped if doing so reduces the distance to ground truth.
    This prevents false negatives caused by left/right confusion in symmetric
    objects.

    Symmetry matching is applied only to samples belonging to
    symmetry_categories. For other categories, no matching is performed.

    Args:
        symmetry_pairs (list[tuple[int, int]]): Pairs of symmetric keypoint
            indices. Default: [(0, 3), (1, 2)].
        symmetry_categories (list[int]): Category IDs eligible for symmetry
            matching. Default: [] (disabled). Set to e.g. [0, 1, 2] to enable.
        **kwargs: Arguments passed to CocoMetric.
    """

    def __init__(self, symmetry_pairs=None, symmetry_categories=None,
                 category_merge=None, **kwargs):
        super().__init__(**kwargs)
        self.symmetry_pairs = symmetry_pairs
        self.symmetry_categories = symmetry_categories
        self.category_merge = category_merge
        self._merge_applied = False
        assert self.symmetry_pairs is not None, "symmetry_pairs must be provided"
        assert self.symmetry_categories is not None, "symmetry_categories must be provided"
        self.symmetry_pairs = np.array(self.symmetry_pairs)

    def process(self, data_batch, data_samples):
        """Process one batch with category-aware symmetry matching."""
        for data_sample in data_samples:
            if self._get_category_id(data_sample) not in self.symmetry_categories:
                continue
            if 'pred_instances' not in data_sample:
                continue

            pred_instances = data_sample['pred_instances']
            if 'keypoints' not in pred_instances:
                continue
            pred_kpts = pred_instances['keypoints']
            if isinstance(pred_kpts, torch.Tensor):
                pred_kpts = pred_kpts.cpu().numpy()

            if len(pred_kpts) == 0:
                continue

            if 'keypoint_scores' in pred_instances:
                pred_scores = pred_instances['keypoint_scores']
                if isinstance(pred_scores, torch.Tensor):
                    pred_scores = pred_scores.cpu().numpy()
            else:
                pred_scores = np.ones(pred_kpts.shape[:2], dtype=np.float32)

            raw_ann = data_sample.get('raw_ann_info', None)
            if raw_ann is None:
                continue

            gt_kpts_list = self._extract_gt_keypoints(raw_ann, pred_kpts.shape[1])
            if not gt_kpts_list:
                continue

            gt_kpts = np.array(gt_kpts_list)
            N, M = len(pred_kpts), len(gt_kpts)
            if N == 0 or M == 0:
                continue

            # Match each pred instance to the closest GT instance
            if M == 1:
                gt_indices = np.zeros(N, dtype=np.int64)
            else:
                dists = np.linalg.norm(
                    pred_kpts[:, :, :2] - gt_kpts[:, None, :, :2], axis=-1
                ).sum(axis=-1)  # (N, M)
                gt_indices = np.argmin(dists, axis=1)  # (N,)

            gt_kpts_matched = gt_kpts[gt_indices]  # (N, K, 2)

            # Vectorized symmetry check: compute all distances at once
            diff = pred_kpts[:, :, :2] - gt_kpts_matched  # (N, K, 2)

            valid = np.max(self.symmetry_pairs, axis=1) < pred_kpts.shape[1]
            if not valid.any():
                continue
            i_idx = self.symmetry_pairs[valid, 0]  # (P',)
            j_idx = self.symmetry_pairs[valid, 1]  # (P',)

            # Original distances: ||pred[i] - gt[i]|| + ||pred[j] - gt[j]||
            dist_orig = (
                np.linalg.norm(diff[:, i_idx], axis=-1) +
                np.linalg.norm(diff[:, j_idx], axis=-1)
            )  # (N, P')

            # Swap distances: ||pred[i] - gt[j]|| + ||pred[j] - gt[i]||
            dist_swap = (
                np.linalg.norm(pred_kpts[:, i_idx, :2] - gt_kpts_matched[:, j_idx, :2], axis=-1) +
                np.linalg.norm(pred_kpts[:, j_idx, :2] - gt_kpts_matched[:, i_idx, :2], axis=-1)
            )  # (N, P')

            take_swap = dist_swap < dist_orig  # (N, P')

            for p_idx, (i, j) in enumerate(zip(i_idx, j_idx)):
                mask = take_swap[:, p_idx]
                if mask.any():
                    pred_kpts[mask, [i, j], :] = pred_kpts[mask, [j, i], :]
                    pred_scores[mask, [i, j]] = pred_scores[mask, [j, i]]

            pred_instances['keypoints'] = pred_kpts
            pred_instances['keypoint_scores'] = pred_scores

        super().process(data_batch, data_samples)

    @staticmethod
    def _get_category_id(data_sample):
        """Extract category ID from a data sample.

        Args:
            data_sample: PoseDataSample object.

        Returns:
            int: Category ID, or -1 if not available.
        """
        # Try direct attribute first
        cid = data_sample.get('category_id', None)
        if cid is not None:
            if isinstance(cid, (np.ndarray, torch.Tensor)):
                cid = cid.item()
            return int(cid)

        # Try gt_instances
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

    @staticmethod
    def _extract_gt_keypoints(raw_ann, num_keypoints):
        """Extract GT keypoints from raw annotation in COCO flat format.

        Args:
            raw_ann: Raw annotation (dict or list of dicts) with 'keypoints'
                field in COCO format [x1, y1, v1, x2, y2, v2, ...].
            num_keypoints: Number of keypoints per instance.

        Returns:
            List of numpy arrays, each of shape (num_keypoints, 2).
        """
        if isinstance(raw_ann, dict):
            raw_ann = [raw_ann]

        gt_kpts_list = []
        for ann in raw_ann:
            if 'keypoints' not in ann:
                continue
            kpts_flat = np.array(ann['keypoints'], dtype=np.float32)
            if len(kpts_flat) >= num_keypoints * 3:
                kpts = kpts_flat[:num_keypoints * 3].reshape(num_keypoints, 3)
                gt_kpts_list.append(kpts[:, :2])

        return gt_kpts_list

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI, with per-category AP.

        Overrides the parent method to add per-category AP metrics
        in addition to the standard COCO metrics.

        Args:
            outfile_prefix (str): The filename prefix of the json files.

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)

        if self.category_merge and not self._merge_applied:
            merge_coco_categories(self.coco, self.category_merge)
            self._merge_applied = True

        sigmas = self.dataset_meta['sigmas']
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas,
                             self.use_area)
        coco_eval.params.useSegm = None

        # Exclude categories with zero GT instances from evaluation
        valid_cat_ids = []
        for cat_id in coco_eval.params.catIds:
            if len(self.coco.getAnnIds(catIds=[cat_id])) > 0:
                valid_cat_ids.append(cat_id)
        if valid_cat_ids:
            coco_eval.params.catIds = valid_cat_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == 'keypoints_crowd':
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)',
                'AP(M)', 'AP(H)'
            ]
        else:
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                'AR .75', 'AR (M)', 'AR (L)'
            ]
        info_str = list(zip(stats_names, coco_eval.stats))

        # --- Per-category AP ---
        precision = coco_eval.eval['precision']
        # precision shape: (T, R, K, A, M)
        # T=10 IoU thresholds, R=101 recall, K=categories, A=4 areas, M=3 maxDets
        cat_ids = coco_eval.params.catIds
        if cat_ids is not None and len(cat_ids) > 0:
            import pandas as pd

            cat_names = {}
            for cat in self.coco.loadCats(cat_ids):
                cat_names[cat['id']] = cat['name']

            recall_levels = np.linspace(0, 1, precision.shape[1])
            scores = coco_eval.eval.get('scores', None)  # (T, R, K, A, M)

            rows = []
            for k, cat_id in enumerate(cat_ids):
                cat_name = cat_names.get(cat_id, f'cat_{cat_id}')
                ap = precision[:, :, k, 0, -1]
                ap = float(np.mean(ap[ap > -1])) if np.any(ap > -1) else 0.0
                ap50 = precision[0, :, k, 0, -1]
                ap50 = float(np.mean(ap50[ap50 > -1])) if np.any(ap50 > -1) else 0.0
                ap75 = precision[5, :, k, 0, -1]
                ap75 = float(np.mean(ap75[ap75 > -1])) if np.any(ap75 > -1) else 0.0

                num_instances = len(self.coco.getAnnIds(catIds=[cat_id]))

                # Per-category best F1 from precision-recall at IoU=0.50
                pr_cat = precision[0, :, k, 0, -1]  # (R,)
                sc_cat = scores[0, :, k, 0, -1] if scores is not None else None  # (R,)
                best_f1 = 0.0
                best_p = 0.0
                best_r = 0.0
                best_thr = 0.0
                for r_idx, r in enumerate(recall_levels):
                    if r == 0 or pr_cat[r_idx] <= -1:
                        continue
                    p = float(pr_cat[r_idx])
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = p
                        best_r = r
                        best_thr = float(sc_cat[r_idx]) if sc_cat is not None else 0.0

                rows.append({
                    'Category': cat_name, 'CategoryID': str(cat_id), 'Instances': num_instances,
                    'AP': ap, 'AP50': ap50, 'AP75': ap75,
                    'F1': best_f1, 'BestP': best_p, 'BestR': best_r, 'BestThr': best_thr
                })
                info_str.append((f'{cat_name}_AP', ap))
                info_str.append((f'{cat_name}_AP50', ap50))
                info_str.append((f'{cat_name}_AP75', ap75))

            df = pd.DataFrame(rows)
            df = df.sort_values('CategoryID')

            # Overall AP over all instances (from coco_eval)
            overall_ap = float(coco_eval.stats[0])
            overall_ap50 = float(coco_eval.stats[1])
            overall_ap75 = float(coco_eval.stats[2])
            total_instances = df['Instances'].sum()

            # Overall best F1: average precision across categories at each recall level
            pr_all = precision[0, :, :, 0, -1]  # (R, K)
            sc_all = scores[0, :, :, 0, -1] if scores is not None else None  # (R, K)
            best_f1 = 0.0
            best_p = 0.0
            best_r = 0.0
            best_thr = 0.0
            for r_idx, r in enumerate(recall_levels):
                if r == 0:
                    continue
                p = np.mean(pr_all[r_idx][pr_all[r_idx] > -1]) if np.any(pr_all[r_idx] > -1) else 0.0
                if p == 0.0:
                    continue
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1 = f1
                    best_p = p
                    best_r = r
                    best_thr = float(np.mean(sc_all[r_idx][sc_all[r_idx] > -1])) if sc_all is not None else 0.0

            avg_row = {
                'Category': 'Avg', 'Instances': total_instances,
                'AP': overall_ap, 'AP50': overall_ap50, 'AP75': overall_ap75,
                'F1': best_f1, 'BestP': best_p, 'BestR': best_r, 'BestThr': best_thr
            }
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            pd.set_option('display.max_colwidth', 20)
            pd.set_option('display.float_format', '{:.4f}'.format)
            logger = MMLogger.get_current_instance()
            logger.info('\n' + df.to_string(index=False))

        return info_str