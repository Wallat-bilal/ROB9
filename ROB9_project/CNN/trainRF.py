# CNN/train_applerf.py

from pathlib import Path
from ultralytics import YOLO

# Root of your project (where main.py lives)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CNN folder (this file lives here)
CNN_ROOT = Path(__file__).resolve().parent

# === USE YOUR NEW APPLE DATASET HERE ===
DATA_YAML = CNN_ROOT / "Data" / "AppleRF" / "data.yaml"

# Where to put YOLO runs for this dataset
RUNS_DIR = PROJECT_ROOT / "runs_AppleRF"   # separate from runs_ssDA


def train_variant(
    model_name: str,
    epochs_phase1: int = 20,
    epochs_phase2: int = 40,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 5e-4,
    img_size: int = 640,
):
    """
    Transfer-learn a YOLOv8 model on your AppleRF dataset in two phases:
    1) freeze backbone, train detection head
    2) unfreeze all layers, fine-tune
    """

    print(f"\n==== Training variant: {model_name} on AppleRF ====\n")

    # --------------------------------------------------
    # Phase 1: load COCO-pretrained weights, freeze backbone
    # --------------------------------------------------
    model = YOLO(model_name)  # e.g. "yolov8s.pt" (COCO-pretrained)

    stem = Path(model_name).stem
    phase1_name = f"{stem}_applerf_phase1"

    print(f"[Phase 1] Training head only for {epochs_phase1} epochs...")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs_phase1,
        imgsz=img_size,
        lr0=lr_phase1,
        freeze=10,              # freeze backbone layers
        project=str(RUNS_DIR),
        name=phase1_name,
        exist_ok=True,
        patience=20,            # early stopping patience (epochs)
        verbose=True,
        device=0,               # use GPU 0
    )

    # Load best weights from Phase 1
    best_phase1 = RUNS_DIR / phase1_name / "weights" / "best.pt"
    print(f"[Phase 1] Best weights saved at: {best_phase1}")

    # --------------------------------------------------
    # Phase 2: fine-tune all layers from phase-1 checkpoint
    # --------------------------------------------------
    model_phase2 = YOLO(str(best_phase1))
    phase2_name = f"{stem}_applerf_phase2"

    print(f"[Phase 2] Fine-tuning ALL layers for {epochs_phase2} epochs...")
    model_phase2.train(
        data=str(DATA_YAML),
        epochs=epochs_phase2,
        imgsz=img_size,
        lr0=lr_phase2,
        freeze=0,               # unfreeze everything
        project=str(RUNS_DIR),
        name=phase2_name,
        exist_ok=True,
        patience=20,
        verbose=True,
        device=0,               # use GPU 0
    )

    best_phase2 = RUNS_DIR / phase2_name / "weights" / "best.pt"
    print(f"[Phase 2] Finished. Final best weights at: {best_phase2}\n")

    return best_phase2


def main():
    # YOLO variants you want to compare on AppleRF
    variants = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    for model_name in variants:
        train_variant(model_name=model_name)


if __name__ == "__main__":
    main()
