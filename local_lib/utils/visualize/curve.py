import matplotlib.pyplot as plt
import numpy as np

# 计算roc和pr曲线的数据点
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

if __name__ == "__main__":
    y_true = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    y_score = np.array(
        [
            0.9,
            0.8,
            0.7,
            0.6,
            0.55,
            0.54,
            0.53,
            0.52,
            0.51,
            0.505,
            0.4,
            0.39,
            0.38,
            0.37,
            0.36,
            0.35,
            0.34,
            0.33,
            0.3,
            0.1,
        ]
    )

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # fpr, tpr, thresholds = roc(y_true, y_score, pos_label=1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="r", linestyle="--", label=f" (roc_auc={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="m", linestyle="--")
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    # plt.savefig("roc_curve.jpg", bbox_inches='tight')
    # plt.show()
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    plt.subplot(1, 2, 2)
    plt.title("PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.step(recall, precision, color="b", label=" (pr_auc={:.4f})".format(pr_auc))
    plt.plot([0, 1], [1, 0], color="m", linestyle="--")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.jpg", bbox_inches="tight")
