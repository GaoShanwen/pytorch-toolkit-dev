import os

from timm.data import RealLabelsImagenet

from local_lib.data.dataset import TxtReaderImageDataset


class RealLabelsCustomData(RealLabelsImagenet):
    def __init__(self, filenames, real_file=None, topk=(1, 5), reader=None):
        if reader is None:
            super(RealLabelsCustomData, self).__init__(filenames, real_file, topk)
            return
        # idx_to_pid = {idx: [int(pid)] for pid, idx in reader.class_to_idx.items()}
        self.filenames, real_labels = [], {}
        for file, idx in reader.samples:
            filename = "/".join(file.split("/")[-2:])
            real_labels.update({filename: [idx]})
            self.filenames.append(filename)
        self.real_labels = real_labels
        assert len(self.filenames) == len(reader.samples)
        self.topk = topk
        self.is_correct = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for pred in pred_batch:
            filename = self.filenames[self.sample_idx]
            # filename = os.path.basename(filename)
            if self.real_labels[filename]:
                for k in self.topk:
                    self.is_correct[k].append(any([p in self.real_labels[filename] for p in pred[:k]]))
            self.sample_idx += 1


if __name__ == "__main__":
    dataset = TxtReaderImageDataset(
        "dataset/zero_600w",
        num_classes=620,
        pass_path="dataset/zero_600w/pass_cats.txt",
        split="val",
        cats_path="dataset/zero_600w/save_cats.txt",
    )
    real_labels = RealLabelsCustomData(dataset.filenames(basename=False), reader=dataset.reader)
