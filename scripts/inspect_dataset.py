from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def file_stats(paths: list[Path]) -> dict[str, float | int]:
    sizes = [path.stat().st_size for path in paths]
    if not sizes:
        return {"count": 0, "total_bytes": 0, "avg_bytes": 0, "min_bytes": 0, "max_bytes": 0}

    return {
        "count": len(sizes),
        "total_bytes": sum(sizes),
        "avg_bytes": sum(sizes) / len(sizes),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
    }


def main() -> None:
    train_rows = read_csv(RAW_DATA_DIR / "train.csv")
    test_rows = read_csv(RAW_DATA_DIR / "test.csv")
    submission_rows = read_csv(RAW_DATA_DIR / "sample_submission.csv")

    train_files = sorted((RAW_DATA_DIR / "train").glob("*.mp4"))
    test_files = sorted((RAW_DATA_DIR / "test").glob("*.mp4"))

    positives = [row for row in train_rows if row["target"] == "1"]
    negatives = [row for row in train_rows if row["target"] == "0"]
    lead_times = [
        float(row["time_of_event"]) - float(row["time_of_alert"])
        for row in positives
        if row["time_of_event"] and row["time_of_alert"]
    ]

    train_ids = {row["id"] for row in train_rows}
    test_ids = {row["id"] for row in test_rows}
    train_file_ids = {path.stem for path in train_files}
    test_file_ids = {path.stem for path in test_files}

    print("Nexar dataset initial inspection")
    print("=" * 34)
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Submission rows: {len(submission_rows)}")
    print(f"Positive train rows: {len(positives)}")
    print(f"Negative train rows: {len(negatives)}")
    print(f"Train videos: {len(train_files)}")
    print(f"Test videos: {len(test_files)}")
    print(f"Missing train videos from CSV: {len(train_ids - train_file_ids)}")
    print(f"Extra train videos not in CSV: {len(train_file_ids - train_ids)}")
    print(f"Missing test videos from CSV: {len(test_ids - test_file_ids)}")
    print(f"Extra test videos not in CSV: {len(test_file_ids - test_ids)}")

    train_stats = file_stats(train_files)
    test_stats = file_stats(test_files)
    print("\nVideo file sizes")
    print(f"Train total GB: {train_stats['total_bytes'] / (1024 ** 3):.2f}")
    print(f"Train average MB: {train_stats['avg_bytes'] / (1024 ** 2):.2f}")
    print(f"Test total GB: {test_stats['total_bytes'] / (1024 ** 3):.2f}")
    print(f"Test average MB: {test_stats['avg_bytes'] / (1024 ** 2):.2f}")

    print("\nPositive event timing")
    print(f"Lead time count: {len(lead_times)}")
    print(f"Lead time min seconds: {min(lead_times):.3f}")
    print(f"Lead time max seconds: {max(lead_times):.3f}")
    print(f"Lead time average seconds: {sum(lead_times) / len(lead_times):.3f}")


if __name__ == "__main__":
    main()
