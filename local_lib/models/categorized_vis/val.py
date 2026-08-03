import torch
from ultralytics.models.yolo.detect import DetectionValidator

class CategorizedVisValidator(DetectionValidator):
    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        orig_plot_matches = self.confusion_matrix.plot_matches
        self.confusion_matrix.plot_matches = lambda img, im_file, save_dir, \
            show_labels=True, show_conf=True: \
            self._categorized_plot_matches(
                orig_plot_matches, img, im_file, save_dir, show_labels, show_conf)

    def _categorized_plot_matches(self, orig_fn, img, im_file, save_dir,
                                  show_labels=True, show_conf=True):
        fp = sum(len(v) for v in self.confusion_matrix.matches["FP"].values())
        fn = sum(len(v) for v in self.confusion_matrix.matches["FN"].values())
        if fp == 0 and fn == 0:
            return
        folder_name = "err" if fp != 0 and fn != 0 else \
            ("fp" if fn == 0 and fp != 0 else "fn")
        orig_fn(img, im_file, save_dir / folder_name, show_labels, show_conf)
