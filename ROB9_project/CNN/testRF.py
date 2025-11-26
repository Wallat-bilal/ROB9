# CNN/test_applerf.py

from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_ROOT = Path(__file__).resolve().parent

# Use your AppleRF dataset yaml
DATA_YAML = CNN_ROOT / "Data" / "AppleRF" / "data.yaml"

# Where the AppleRF training runs were saved (from train_applerf.py)
RUNS_DIR = PROJECT_ROOT / "runs_AppleRF"


def eval_variant(model_stem: str, conf: float = 0.001, iou: float = 0.5, split: str = "val"):
    """
    Evaluate a trained YOLOv8 model (phase 2 best checkpoint) on the AppleRF dataset.

    Args:
        model_stem: e.g. "yolov8s" (must match what you used in train_applerf.py)
        conf: confidence threshold for evaluation
        iou: IoU threshold for NMS / metrics
        split: which split to evaluate on ("val" or "test")
    """
    phase2_name = f"{model_stem}_applerf_phase2"
    best_weights = RUNS_DIR / phase2_name / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"[WARN] Weights not found for {phase2_name}: {best_weights}")
        return

    print(f"\n==== Evaluating {best_weights} ====\n")

    model = YOLO(str(best_weights))

    results = model.val(
        data=str(DATA_YAML),
        split=split,    # "val" or "test"
        conf=conf,
        iou=iou,
        verbose=True,
    )

    metrics = results.results_dict
    print("Raw metrics dict:")
    print(metrics)

    p = metrics.get("metrics/precision(B)", None)
    r = metrics.get("metrics/recall(B)", None)
    map50 = metrics.get("metrics/mAP50(B)", None)
    map5095 = metrics.get("metrics/mAP50-95(B)", None)

    if p is not None and r is not None:
        f1 = 2 * p * r / (p + r + 1e-16)
    else:
        f1 = None

    print("\nSummary:")
    print(f" Precision: {p:.4f}" if p is not None else " Precision: N/A")
    print(f" Recall   : {r:.4f}" if r is not None else " Recall   : N/A")
    print(f" F1-score : {f1:.4f}" if f1 is not None else " F1-score : N/A")
    print(f" mAP@0.50 : {map50:.4f}" if map50 is not None else " mAP@0.50 : N/A")
    print(f" mAP@0.50-0.95: {map5095:.4f}" if map5095 is not None else " mAP@0.50-0.95: N/A")


def main():
    # Must match the stems used in train_applerf.py (without .pt)
    model_stems = ["yolov8n", "yolov8s", "yolov8m"]

    for stem in model_stems:
        # you can switch to split="test" later if you want final test metrics
        eval_variant(stem, split="val")


if __name__ == "__main__":
    main()
