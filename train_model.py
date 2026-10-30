from __future__ import annotations

import argparse
from hashlib import sha256
import math
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_CONFIG = DATA_DIR / "data.yaml"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_OUTPUT = ROOT / "models" / "brain_tumor_yolo_best.pt"


def _validate_label(label_path: Path) -> set[int]:
    classes = set()
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = line.split()
        if len(values) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected class and four box values"
            )

        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(
                f"{label_path}:{line_number}: invalid numeric annotation"
            ) from error

        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(
                f"{label_path}:{line_number}: class id {class_id} is out of range"
            )
        if any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in coordinates
        ):
            raise ValueError(
                f"{label_path}:{line_number}: box values must be between 0 and 1"
            )
        if coordinates[2] == 0 or coordinates[3] == 0:
            raise ValueError(
                f"{label_path}:{line_number}: box width and height must be positive"
            )

        classes.add(class_id)

    return classes


def _image_hash(image_path: Path) -> str:
    digest = sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_image_manifest(
    split: str,
    allow_missing_labels: bool,
    excluded_hashes: set[str] | None = None,
) -> tuple[Path, set[int], dict[str, int], set[str]]:
    split_dir = DATA_DIR / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image and label directories for {split}: "
            f"{image_dir} and {label_dir}"
        )

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported {split} images found in {image_dir}")

    image_groups = {}
    missing_labels = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(image_path)
            continue
        classes = _validate_label(label_path)
        annotation = tuple(
            sorted(
                " ".join(line.split())
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        )
        image_groups.setdefault(_image_hash(image_path), []).append(
            (image_path, classes, annotation)
        )

    if missing_labels and not allow_missing_labels:
        examples = ", ".join(str(path) for path in missing_labels[:5])
        raise ValueError(
            f"{split} images are missing labels ({len(missing_labels)}): {examples}"
        )
    selected_images = []
    classes = set()
    stats = {
        "missing_labels": len(missing_labels),
        "duplicate_images": 0,
        "overlapping_images": 0,
        "conflicting_images": 0,
        "box_disagreement_groups": 0,
    }

    for image_hash, records in sorted(image_groups.items()):
        if excluded_hashes and image_hash in excluded_hashes:
            stats["overlapping_images"] += len(records)
            continue

        class_sets = {frozenset(record[1]) for record in records}
        if len(class_sets) > 1:
            stats["conflicting_images"] += len(records)
            continue

        annotations = {record[2] for record in records}
        if split == "val" and len(annotations) > 1:
            stats["conflicting_images"] += len(records)
            continue

        if len(records) > 1:
            stats["duplicate_images"] += len(records) - 1
        if len(annotations) > 1:
            stats["box_disagreement_groups"] += 1

        image_path, image_classes, _ = records[0]
        selected_images.append(image_path)
        classes.update(image_classes)

    if not selected_images:
        raise ValueError(f"No usable {split} images found in {image_dir}")

    manifest_path = split_dir / "annotated_images.txt"
    manifest_path.write_text(
        "".join(f"{path.resolve()}\n" for path in selected_images),
        encoding="utf-8",
    )
    return manifest_path, classes, stats, set(image_groups)


def prepare_dataset() -> None:
    train_manifest, train_classes, train_stats, train_hashes = _create_image_manifest(
        "train", allow_missing_labels=False
    )
    val_manifest, val_classes, val_stats, _ = _create_image_manifest(
        "val", allow_missing_labels=True, excluded_hashes=train_hashes
    )
    expected_classes = set(range(len(CLASS_NAMES)))
    if train_classes != expected_classes:
        raise ValueError(
            f"Training labels must cover all classes 0-{len(CLASS_NAMES) - 1}; "
            f"found {sorted(train_classes)}"
        )
    if val_classes != expected_classes:
        raise ValueError(
            f"Leak-free validation labels must cover all classes 0-{len(CLASS_NAMES) - 1}; "
            f"found {sorted(val_classes)}"
        )

    print(f"Training images: {len(train_manifest.read_text(encoding='utf-8').splitlines())}")
    print(f"Validation images: {len(val_manifest.read_text(encoding='utf-8').splitlines())}")
    if train_stats["duplicate_images"]:
        print(
            "Training duplicates removed: "
            f"{train_stats['duplicate_images']} duplicate files."
        )
    if train_stats["conflicting_images"]:
        print(
            "Warning: excluded training images with conflicting class labels: "
            f"{train_stats['conflicting_images']}."
        )
    if train_stats["box_disagreement_groups"]:
        print(
            "Warning: in same-class duplicate training groups with differing boxes, "
            "the first sorted annotation was retained: "
            f"{train_stats['box_disagreement_groups']} groups."
        )
    if val_stats["missing_labels"]:
        print(
            f"Warning: excluded {val_stats['missing_labels']} validation images without "
            "matching labels."
        )
    if val_stats["overlapping_images"]:
        print(
            "Validation duplicates of training images excluded: "
            f"{val_stats['overlapping_images']}."
        )
    if val_stats["duplicate_images"] or val_stats["conflicting_images"]:
        print(
            "Validation duplicate/conflicting images excluded: "
            f"{val_stats['duplicate_images'] + val_stats['conflicting_images']}."
        )


def train(epochs: int, batch_size: int) -> None:
    prepare_dataset()

    model = YOLO(str(ROOT / "yolov8n.pt"))
    model.train(
        data=str(DATA_CONFIG),
        epochs=epochs,
        patience=20,
        batch=batch_size,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        project=str(ROOT / "runs" / "detect"),
        name="brain_tumor_yolo",
        exist_ok=False,
        pretrained=True,
        seed=42,
        deterministic=True,
        cache=False,
        close_mosaic=10,
        val=True,
        plots=True,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training completed without best weights: {best_weights}")

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = MODEL_OUTPUT.with_suffix(".pt.tmp")
    shutil.copy2(best_weights, temporary_output)
    os.replace(temporary_output, MODEL_OUTPUT)
    print(f"Best weights saved to: {MODEL_OUTPUT}")
    print(f"Training run saved to: {model.trainer.save_dir}")
    print(f"Validation metrics: {model.trainer.metrics}")


def activate_existing_checkpoint(checkpoint: Path) -> None:
    prepare_dataset()

    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = YOLO(str(checkpoint))
    if len(model.names) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected a {len(CLASS_NAMES)}-class checkpoint, found {len(model.names)} classes"
        )
    model.model.names = dict(enumerate(CLASS_NAMES))
    metrics = model.val(
        data=str(DATA_CONFIG),
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        plots=True,
    )

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = MODEL_OUTPUT.with_suffix(".pt.tmp")
    model.save(temporary_output)
    os.replace(temporary_output, MODEL_OUTPUT)
    print(f"Corrected checkpoint saved to: {MODEL_OUTPUT}")
    print(f"Validation metrics: {metrics.results_dict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or validate a brain-tumor YOLO detector."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--existing-checkpoint",
        type=Path,
        help="Correct class names and validate an existing checkpoint instead of training.",
    )
    args = parser.parse_args()

    if args.existing_checkpoint is not None:
        activate_existing_checkpoint(args.existing_checkpoint)
        return
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    train(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
