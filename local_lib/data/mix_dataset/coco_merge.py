"""COCO annotation category merging utilities.

Provides functions to merge multiple categories into one in COCO-format
annotations, and a data pipeline transform to remap category IDs during
training and validation.
"""

import json
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
from mmpose.registry import TRANSFORMS


def build_merge_mapping(merge_groups: List[List[str]],
                        categories: List[dict]) -> Dict[int, int]:
    """Build a category ID remapping from merge groups.

    Args:
        merge_groups: List of merge groups, e.g.,
            ``[["Oven_TopHandle", "Oven_BottomHandle"], ["Oven_TopInner", "Oven_BottomInner"]]``.
            The first name in each group is the source (to be replaced);
            the second is the target (to keep).
        categories: COCO-format categories list, each dict has ``id`` and
            ``name`` keys.

    Returns:
        dict: Mapping from old category ID to new category ID, i.e.,
            ``{old_cat_id: new_cat_id}``.
    """
    name_to_id = {cat['name']: cat['id'] for cat in categories}

    id_mapping = OrderedDict()
    for group in merge_groups:
        if len(group) < 2:
            continue
        source_name = group[0]
        target_name = group[1]
        if source_name not in name_to_id or target_name not in name_to_id:
            continue
        source_id = name_to_id[source_name]
        target_id = name_to_id[target_name]
        id_mapping[source_id] = target_id

    return id_mapping


def build_merge_mapping_from_file(
        merge_groups: List[List[str]],
        ann_file: str) -> Dict[int, int]:
    """Build a category ID remapping from an annotation file.

    Args:
        merge_groups: Same as :func:`build_merge_mapping`.
        ann_file: Path to COCO annotation JSON file.

    Returns:
        dict: Mapping from old category ID to new category ID.
    """
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    return build_merge_mapping(merge_groups, coco_data['categories'])


def merge_coco_categories(coco, merge_groups: List[List[str]]) -> Dict[int, int]:
    """Merge categories in a COCO object in-place.

    Each group in merge_groups specifies a pair of categories to merge; the
    first category name is the source (to be replaced), and the second is
    the target (to keep).

    Args:
        coco: COCO object (pycocotools.coco.COCO).
        merge_groups: List of merge groups, e.g.,
            ``[["Oven_TopHandle", "Oven_BottomHandle"]]``.

    Returns:
        dict: Mapping from old category IDs to new category IDs,
            ``{old_cat_id: new_cat_id}``.
    """
    id_mapping = build_merge_mapping(merge_groups, coco.dataset['categories'])

    if not id_mapping:
        return id_mapping

    merged_ids = set(id_mapping.keys())
    coco.dataset['categories'] = [
        cat for cat in coco.dataset['categories']
        if cat['id'] not in merged_ids
    ]

    for ann in coco.dataset['annotations']:
        if ann['category_id'] in id_mapping:
            ann['category_id'] = id_mapping[ann['category_id']]

    coco.createIndex()

    return id_mapping


@TRANSFORMS.register_module()
class MergeCategory:
    """Data pipeline transform to remap category IDs during training/validation.

    This transform should be placed after ``LoadAnnotations`` in the pipeline.
    It remaps the ``category_id`` field in the results dict according to the
    merge mapping.

    The original category IDs (channels) are preserved in the model output;
    only the annotation labels are remapped. This ensures the model's output
    dimension stays consistent with the pre-merge configuration.

    Args:
        merge_groups (list[list[str]]): List of merge groups. The first name
            in each group is the source (to be replaced); the second is
            the target (to keep).
            E.g., ``[["Oven_TopHandle", "Oven_BottomHandle"]]`` replaces
            Oven_TopHandle with Oven_BottomHandle.
        ann_file (str, optional): Path to COCO annotation JSON file. Used to
            resolve category names to IDs. If not provided, will try to
            resolve from ``results['ann_file']`` or ``results['dataset_meta']``.
        ignore_missing (bool): Whether to silently ignore missing categories.
            Default: True.
    """

    def __init__(self,
                 merge_groups: List[List[str]],
                 ann_file: Optional[str] = None,
                 ignore_missing: bool = True):
        self.merge_groups = merge_groups
        self.ann_file = ann_file
        self.ignore_missing = ignore_missing
        self._id_mapping: Optional[Dict[int, int]] = None

    def _lazy_init(self, results: dict):
        """Resolve category names to IDs on first call."""
        if self._id_mapping is not None:
            return

        categories = None
        if self.ann_file:
            with open(self.ann_file, 'r') as f:
                categories = json.load(f).get('categories', [])
        elif 'dataset_meta' in results:
            categories = results['dataset_meta'].get('categories', [])
        elif 'ann_info' in results and 'category_id' in results:
            # Fallback: cannot resolve names, skip
            self._id_mapping = {}
            return

        if categories:
            self._id_mapping = build_merge_mapping(self.merge_groups, categories)
        else:
            self._id_mapping = {}

    def __call__(self, results: dict) -> dict:
        self._lazy_init(results)

        if not self._id_mapping:
            return results

        if 'category_id' in results:
            cid = results['category_id']
            if isinstance(cid, np.ndarray):
                cid = cid.item()
            if cid in self._id_mapping:
                results['category_id'] = self._id_mapping[cid]

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'merge_groups={self.merge_groups})')