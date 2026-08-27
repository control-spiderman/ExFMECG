"""Map record-level ICD codes to PheCodeX using the official flat maps."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnoses", required=True, type=Path)
    parser.add_argument("--cm-map", required=True, type=Path)
    parser.add_argument("--who-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record-column", default="record_id")
    parser.add_argument("--code-column", default="icd_code")
    parser.add_argument("--vocabulary-column", default="vocabulary_id")
    return parser.parse_args()


def normalize_code(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def normalize_vocabulary(value):
    value = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    aliases = {
        "9": "ICD9CM",
        "10": "ICD10CM",
        "ICD9": "ICD9CM",
        "ICD10CM": "ICD10CM",
        "ICD10": "ICD10",
    }
    return aliases.get(value, value)


def load_map(path, code_candidates):
    mapping = defaultdict(list)
    with path.open(encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle)
        code_column = next(
            (name for name in code_candidates if name in reader.fieldnames), None
        )
        if code_column is None or "vocabulary_id" not in reader.fieldnames:
            raise ValueError(f"Unrecognized PheCodeX mapping columns in {path}")
        for row in reader:
            key = (
                normalize_vocabulary(row["vocabulary_id"]),
                normalize_code(row[code_column]),
            )
            mapping[key].append(
                {
                    "phecode": row["phecode"],
                    "phecode_string": row["phecode_string"],
                    "category": row.get("category", ""),
                }
            )
    return mapping


def main():
    args = parse_args()
    mapping = load_map(args.cm_map, ("ICD", "icd"))
    for key, values in load_map(args.who_map, ("icd", "ICD")).items():
        mapping[key].extend(values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapped_rows = 0
    with (
        args.diagnoses.open(encoding="utf-8-sig", newline="") as source,
        args.output.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        required = {args.record_column, args.code_column, args.vocabulary_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing diagnosis columns: {sorted(missing)}")
        fieldnames = [
            args.record_column,
            "icd_code",
            "vocabulary_id",
            "phecode",
            "phecode_string",
            "category",
        ]
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            vocabulary = normalize_vocabulary(row[args.vocabulary_column])
            code = normalize_code(row[args.code_column])
            seen = set()
            for match in mapping.get((vocabulary, code), []):
                identity = (match["phecode"], match["phecode_string"])
                if identity in seen:
                    continue
                seen.add(identity)
                writer.writerow(
                    {
                        args.record_column: row[args.record_column],
                        "icd_code": row[args.code_column],
                        "vocabulary_id": vocabulary,
                        **match,
                    }
                )
                mapped_rows += 1
    print(f"Wrote {mapped_rows} mapped diagnosis rows to {args.output}")


if __name__ == "__main__":
    main()
