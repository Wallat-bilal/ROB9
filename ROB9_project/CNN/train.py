# CNN/train.py

from pathlib import Path
from ultralytics import YOLO


# Root of your project (where main.py lives)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CNN folder (this file lives here)
CNN_ROOT = Path(__file__).resolve().parent
# Dataset yaml
DATA_YAML = CNN_ROOT / "ssda.yaml"

# Where to put YOLO runs
RUNS_DIR = PROJECT_ROOT / "runs_ssDA"   # note: folder name is up to you


def train_variant(
    model_name: str,
    epochs_phase1: int = 20,
    epochs_phase2: int = 40,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 5e-4,
    img_size: int = 640,
):
    """
    Transfer-learn a YOLOv8 model on SSDA in two phases:
    1) freeze backbone, train detection head
    2) unfreeze all layers, fine-tune
    """

    print(f"\n==== Training variant: {model_name} ====\n")

    # --------------------------------------------------
    # Phase 1: load COCO-pretrained weights, freeze backbone
    # --------------------------------------------------
    model = YOLO(model_name)  # e.g. "yolov8n.pt" (COCO-pretrained)

    phase1_name = f"{Path(model_name).stem}_ssda_phase1"

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
    )

    # Load best weights from Phase 1
    best_phase1 = RUNS_DIR / phase1_name / "weights" / "best.pt"
    print(f"[Phase 1] Best weights saved at: {best_phase1}")

    # --------------------------------------------------
    # Phase 2: fine-tune all layers from phase-1 checkpoint
    # --------------------------------------------------
    model_phase2 = YOLO(str(best_phase1))

    phase2_name = f"{Path(model_name).stem}_ssda_phase2"

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
    )

    best_phase2 = RUNS_DIR / phase2_name / "weights" / "best.pt"
    print(f"[Phase 2] Finished. Final best weights at: {best_phase2}\n")

    return best_phase2


def main():
    # List of YOLO variants you want to compare
    variants = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    for model_name in variants:
        train_variant(model_name=model_name)


if __name__ == "__main__":
    main()
