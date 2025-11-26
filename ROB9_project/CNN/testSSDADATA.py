# CNN/test.py

from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_ROOT = Path(__file__).resolve().parent
DATA_YAML = CNN_ROOT / "ssda.yaml"
RUNS_DIR = PROJECT_ROOT / "runs_ssDA"


def eval_variant(model_stem: str, conf: float = 0.001, iou: float = 0.5):
    """
    Evaluate a trained YOLOv8 model (phase 2 best checkpoint) on SSDA val set.
    Prints precision, recall, F1-score, mAP50, mAP50-95.
    """
    phase2_name = f"{model_stem}_ssda_phase2"
    best_weights = RUNS_DIR / phase2_name / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"[WARN] Weights not found for {phase2_name}: {best_weights}")
        return

    print(f"\n==== Evaluating {best_weights} ====\n")

    model = YOLO(str(best_weights))

    # Run validation
    # model.val() returns an object with metrics and curves
    results = model.val(
        data=str(DATA_YAML),
        split="val",
        conf=conf,
        iou=iou,
        verbose=True,
    )

    metrics = results.results_dict  # dict of metrics (P, R, mAP, etc.)
    print("Raw metrics dict:")
    print(metrics)

    # Extract overall (box) precision and recall
    # Keys are typically like 'metrics/precision(B)' and 'metrics/recall(B)'
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

    # YOLO also saves curves like F1_curve.png, PR_curve.png, etc. in the run folder. :contentReference[oaicite:3]{index=3}


def main():
    # Must match the stems used in train.py (without .pt)
    model_stems = ["yolov8n", "yolov8s", "yolov8m"]

    for stem in model_stems:
        eval_variant(stem)


if __name__ == "__main__":
    main()
