from typing import Any

from pytorch_lightning import LightningDataModule

from rfdetr.utilities.logger import get_logger

from .dataset import MixedRFDETRDataset

logger = get_logger(__name__)


class MixedRFDETRDataModule(LightningDataModule):
    """RFDETRDataModule subclass that wraps the train split with class remapping
    and per-epoch background mixing.

    This wrapper delegates all functionality to the provided base datamodule but
    ensures the mixed training dataset is injected before the base.setup("fit")
    call so DataLoaders are built from the mixed dataset. It subclasses
    LightningDataModule so PyTorch Lightning's `is_overridden` checks work as
    expected.
    """

    def __init__(
        self,
        base_datamodule: Any,
        mixed_train: MixedRFDETRDataset,
    ):
        self._base = base_datamodule
        self._mixed_train = mixed_train

    def __getattr__(self, name: str):
        """Forward all unknown attributes to the base datamodule."""
        return getattr(self._base, name)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            try:
                self._base._dataset_train = self._mixed_train
            except Exception:
                pass

        self._base.setup(stage)

        if stage == "fit":
            try:
                if self._mixed_train.class_mapping:
                    base_val = getattr(self._base, "_dataset_val", None)
                    if base_val is not None and not isinstance(base_val, MixedRFDETRDataset):
                        self._base._dataset_val = MixedRFDETRDataset(
                            base_dataset=base_val,
                            class_mapping=self._mixed_train.class_mapping,
                            background_im_files=None,
                            alpha=None,
                        )
                    elif base_val is None:
                        raise AssertionError("Base datamodule did not expose a validation dataset to wrap; cannot apply class mapping")
            except Exception:
                logger.exception("Could not wrap val dataset with class_mapping; failing")

            logger.info(
                "Train split wrapped: main=%d, bg=%d (alpha=%s)",
                self._mixed_train._n_main,
                self._mixed_train._n_bg,
                self._mixed_train.alpha,
            )

    def prepare_data(self) -> None:
        fn = getattr(self._base, "prepare_data", None)
        if callable(fn):
            return fn()

    def train_dataloader(self, *args, **kwargs):
        fn = getattr(self._base, "train_dataloader", None)
        if callable(fn):
            return fn(*args, **kwargs)
        raise RuntimeError("Underlying datamodule has no train_dataloader")

    def val_dataloader(self, *args, **kwargs):
        fn = getattr(self._base, "val_dataloader", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return None

    def test_dataloader(self, *args, **kwargs):
        fn = getattr(self._base, "test_dataloader", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return None

    def predict_dataloader(self, *args, **kwargs):
        fn = getattr(self._base, "predict_dataloader", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return None