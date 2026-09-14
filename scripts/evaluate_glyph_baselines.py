from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from so101_typing.perception.glyph_baselines import (
    HOGLinearSVMGlyphClassifier,
    TemplateGlyphClassifier,
)


def resolve_processed_image(dataset_root: Path, item: dict) -> Path:
    recorded = Path(str(item["processed_glyph"]))
    if recorded.exists():
        return recorded

    fallback = (
        dataset_root
        / "by_label"
        / str(item["label"])
        / recorded.name
    )
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Cannot resolve processed glyph for {item['label']} "
        f"{item['sample']} candidate {item['candidate_index']}"
    )


def load_dataset(dataset_root: Path) -> list[dict]:
    payload = json.loads(
        (dataset_root / "dataset.json").read_text(encoding="utf-8")
    )
    rows = []
    for item in payload["items"]:
        path = resolve_processed_image(dataset_root, item)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read {path}")
        rows.append({**item, "image": image, "resolved_path": str(path)})
    return rows


def make_error_sheet(errors: list[dict], output: Path) -> None:
    tile = 96
    columns = 8
    rows = max(1, (len(errors) + columns - 1) // columns)
    sheet = np.full((rows * tile, columns * tile, 3), 255, np.uint8)

    for i, item in enumerate(errors):
        row, col = divmod(i, columns)
        image = cv2.cvtColor(item["image"], cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
        x = col * tile + 16
        y = row * tile + 20
        sheet[y : y + 64, x : x + 64] = image
        text = f"{item['label']}->{item['prediction']}"
        cv2.putText(
            sheet,
            text,
            (col * tile + 3, row * tile + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output), sheet)


def evaluate_method(rows: list[dict], method: str) -> dict:
    groups = sorted({str(row["sample"]) for row in rows})
    predictions = []
    skipped = []
    folds = []

    for held_out in groups:
        train = [row for row in rows if row["sample"] != held_out]
        test = [row for row in rows if row["sample"] == held_out]
        train_labels = {str(row["label"]) for row in train}
        eligible = [row for row in test if row["label"] in train_labels]

        labels = [str(row["label"]) for row in train]
        images = [row["image"] for row in train]

        if method == "template":
            classifier = TemplateGlyphClassifier(max_shift=2)
            classifier.fit(labels, images)
        elif method == "hog_svm":
            classifier = HOGLinearSVMGlyphClassifier()
            classifier.fit(labels, images, augment=True)
        else:
            raise ValueError(method)

        correct = 0
        for row in test:
            if row["label"] not in train_labels:
                skipped.append(
                    {
                        "sample": row["sample"],
                        "label": row["label"],
                        "candidate_index": row["candidate_index"],
                        "reason": "class_absent_from_training_fold",
                    }
                )
                continue

            prediction = classifier.predict(row["image"])
            is_correct = prediction == row["label"]
            correct += int(is_correct)
            predictions.append(
                {
                    "sample": row["sample"],
                    "label": row["label"],
                    "prediction": prediction,
                    "candidate_index": row["candidate_index"],
                    "correct": is_correct,
                    "image": row["image"],
                }
            )

        folds.append(
            {
                "held_out_sample": held_out,
                "eligible": len(eligible),
                "correct": correct,
                "accuracy": correct / len(eligible) if eligible else None,
            }
        )

    total = len(predictions)
    correct = sum(int(row["correct"]) for row in predictions)
    confusion = Counter(
        (row["label"], row["prediction"])
        for row in predictions
    )
    return {
        "method": method,
        "eligible": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "skipped": skipped,
        "folds": folds,
        "predictions": predictions,
        "confusion": confusion,
    }


def serializable(result: dict) -> dict:
    return {
        "method": result["method"],
        "eligible": result["eligible"],
        "correct": result["correct"],
        "accuracy": result["accuracy"],
        "skipped": result["skipped"],
        "folds": result["folds"],
        "errors": [
            {
                k: row[k]
                for k in (
                    "sample",
                    "label",
                    "prediction",
                    "candidate_index",
                )
            }
            for row in result["predictions"]
            if not row["correct"]
        ],
    }


def write_confusion_csv(result: dict, path: Path) -> None:
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for truth in labels:
            writer.writerow(
                [
                    truth,
                    *[
                        result["confusion"].get((truth, pred), 0)
                        for pred in labels
                    ],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped real-frame evaluation for Phase-2 glyph baselines."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_glyph_baseline_eval"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("template", "hog_svm"),
        default=("template", "hog_svm"),
        help="Baselines to run. Use --methods hog_svm for the larger round-2 dataset.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(args.dataset_root)
    results = {
        method: evaluate_method(rows, method)
        for method in args.methods
    }

    payload = {
        "protocol": (
            "leave-one-source-frame-out; test samples whose class is absent "
            "from the training fold are reported as skipped"
        ),
        "dataset_count": len(rows),
        "results": {
            name: serializable(result)
            for name, result in results.items()
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    for name, result in results.items():
        write_confusion_csv(
            result,
            args.output / f"confusion_{name}.csv",
        )
        errors = [
            row for row in result["predictions"] if not row["correct"]
        ]
        make_error_sheet(
            errors,
            args.output / f"errors_{name}.jpg",
        )

    print(f"[OK] dataset: {len(rows)} glyphs")
    for name, result in results.items():
        print(
            f"[OK] {name}: "
            f"{result['correct']}/{result['eligible']} "
            f"= {100.0 * result['accuracy']:.1f}% "
            f"(skipped {len(result['skipped'])})"
        )
        for fold in result["folds"]:
            print(
                f"  {fold['held_out_sample']}: "
                f"{fold['correct']}/{fold['eligible']} "
                f"= {100.0 * fold['accuracy']:.1f}%"
            )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
