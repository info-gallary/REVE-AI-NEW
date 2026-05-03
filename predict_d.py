import torch
from torchvision import models, transforms
from PIL import Image
from typing import Optional, Sequence
import os

CLASS_NAMES = [
    "Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives", "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections"
]


def load_model(weights_path, num_classes):
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_d(image: Image.Image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(r"./models/model_epoch_25.pth", num_classes=23)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    return {"class": CLASS_NAMES[predicted_class], "confidence": confidence}


def create_evaluation_graphs(
    y_true,
    y_prob,
    class_names: Optional[Sequence[str]] = None,
    out_dir: str = "evaluation_d",
    prefix: str = "d",
):
    """
    Generate and save evaluation graphs (confusion matrix, ROC, PR, and per-class metrics)
    for multi-class classification. This is designed for research paper figures.

    Parameters
    - y_true: 1D array-like of shape (n_samples,) with integer class ids
    - y_prob: 2D array-like of shape (n_samples, n_classes) with probabilities for each class
    - class_names: list of class names; defaults to this module's CLASS_NAMES
    - out_dir: directory to save figures
    - prefix: filename prefix for saved figures

    Saves
    - {prefix}_confusion_matrix.png (counts)
    - {prefix}_confusion_matrix_normalized.png (row-normalized)
    - {prefix}_roc_curves.png (micro, macro, per-class)
    - {prefix}_pr_curves.png (micro-average PR)
    - {prefix}_per_class_metrics.png (precision/recall/F1 bars)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
        classification_report,
    )
    from sklearn.preprocessing import label_binarize

    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array of class indices")
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be 2D array of probabilities [n_samples, n_classes]")
    n_classes = y_prob.shape[1]
    if class_names is None:
        class_names = CLASS_NAMES
    if len(class_names) != n_classes:
        raise ValueError(f"class_names length {len(class_names)} != n_classes {n_classes}")

    # Confusion matrix (counts)
    cm = confusion_matrix(y_true, np.argmax(y_prob, axis=1), labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * n_classes, 1.0 + 0.5 * n_classes), dpi=160)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title("Confusion Matrix (Counts)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plt.close(fig)

    # Confusion matrix (row-normalized)
    with np.errstate(all="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * n_classes, 1.0 + 0.5 * n_classes), dpi=160)
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45, colorbar=True, values_format=".2f")
    ax.set_title("Confusion Matrix (Row-Normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix_normalized.png"))
    plt.close(fig)

    # ROC curves (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC = {roc_auc['micro']:.3f})", color="deeppink", linewidth=2)
    ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC = {roc_auc['macro']:.3f})", color="navy", linewidth=2)
    # Plot per-class lightly to avoid clutter; for 23 classes this will be dense
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=1, alpha=0.5, label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (OvR)")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_roc_curves.png"))
    plt.close(fig)

    # Precision-Recall curves (micro-average + per class)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    pr_auc_micro = average_precision_score(y_true_bin, y_prob, average="micro")
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
    ax.plot(recall, precision, color="darkorange", linewidth=2, label=f"micro-average (AP={pr_auc_micro:.3f})")
    for i in range(n_classes):
        pi, ri, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap_i = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(ri, pi, lw=1, alpha=0.5, label=f"{class_names[i]} (AP={ap_i:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (OvR)")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_pr_curves.png"))
    plt.close(fig)

    # Per-class metrics (precision, recall, f1)
    report = classification_report(y_true, np.argmax(y_prob, axis=1), target_names=class_names, output_dict=True, zero_division=0)
    precisions = [report[c]["precision"] for c in class_names]
    recalls = [report[c]["recall"] for c in class_names]
    f1s = [report[c]["f1-score"] for c in class_names]

    import numpy as np
    x = np.arange(n_classes)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * n_classes), 5), dpi=160)
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=60, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_per_class_metrics.png"))
    plt.close(fig)

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "ap_micro": pr_auc_micro,
    }

