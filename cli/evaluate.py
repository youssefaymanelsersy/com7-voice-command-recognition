"""Offline evaluation for saved feature datasets."""

import argparse
import json
from collections import Counter, defaultdict

from core.recognize import recognize_features


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_model(model_path="models/model.json", dataset_path="data/test_samples.json"):
    model = _load_json(model_path)
    dataset = _load_json(dataset_path)

    commands = list(dataset.keys())
    rows = []
    confusion = defaultdict(Counter)
    total = 0
    correct = 0
    unknown = 0

    for expected in commands:
        predictions = []
        command_correct = 0
        command_unknown = 0

        for sample in dataset.get(expected, []):
            prediction = recognize_features(sample, model)
            predictions.append(prediction)
            confusion[expected][prediction] += 1

            total += 1
            if prediction == expected:
                correct += 1
                command_correct += 1
            if prediction == "unknown":
                unknown += 1
                command_unknown += 1

        count = len(predictions)
        rows.append(
            {
                "command": expected,
                "count": count,
                "correct": command_correct,
                "unknown": command_unknown,
                "accuracy": (command_correct / count) if count else 0.0,
                "predictions": predictions,
            }
        )

    accuracy = (correct / total) if total else 0.0
    unknown_rate = (unknown / total) if total else 0.0
    return {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "commands": commands,
        "total": total,
        "correct": correct,
        "unknown": unknown,
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "rows": rows,
        "confusion": confusion,
    }


def print_report(report):
    print(f"Model: {report['model_path']}")
    print(f"Dataset: {report['dataset_path']}")
    print(
        f"Overall: {report['correct']}/{report['total']} correct "
        f"({report['accuracy'] * 100:.1f}%), "
        f"unknown={report['unknown']} ({report['unknown_rate'] * 100:.1f}%)"
    )
    print()

    print("Per command:")
    for row in report["rows"]:
        print(
            f"  {row['command']:<6} "
            f"{row['correct']:>2}/{row['count']:<2} "
            f"accuracy={row['accuracy'] * 100:>5.1f}% "
            f"unknown={row['unknown']:<2} "
            f"predictions={', '.join(row['predictions'])}"
        )

    labels = sorted(
        set(report["commands"])
        | {"unknown"}
        | {pred for counts in report["confusion"].values() for pred in counts}
    )

    print()
    print("Confusion matrix:")
    print("expected \\ predicted".ljust(20) + "".join(label[:7].rjust(8) for label in labels))
    for expected in report["commands"]:
        counts = report["confusion"][expected]
        print(expected.ljust(20) + "".join(str(counts[label]).rjust(8) for label in labels))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved voice command model offline.")
    parser.add_argument("--model", default="models/model.json", help="Path to model JSON.")
    parser.add_argument(
        "--dataset",
        default="data/test_samples.json",
        help="Path to saved feature dataset.",
    )
    args = parser.parse_args()

    print_report(evaluate_model(args.model, args.dataset))


if __name__ == "__main__":
    main()
