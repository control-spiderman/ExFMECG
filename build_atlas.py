"""Build disease-ECG concept relationships from held-out cohort exports."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, rankdata


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    eligibility = subparsers.add_parser(
        "eligibility", help="Select diseases eligible for interpretability analysis"
    )
    eligibility.add_argument("--performance", nargs="+", required=True, type=Path)
    eligibility.add_argument("--output", required=True, type=Path)

    rank = subparsers.add_parser(
        "rank", help="Convert centre-level attribution to top-50 candidates"
    )
    rank.add_argument("--attribution", nargs="+", required=True, type=Path)
    rank.add_argument("--output-dir", required=True, type=Path)
    rank.add_argument("--top-k", type=int, default=50)

    calibrate = subparsers.add_parser(
        "calibrate", help="Calibrate binary concept thresholds by maximum MCC"
    )
    calibrate.add_argument("--phenotype", nargs="+", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser(
        "validate", help="Validate candidate relationships across centres"
    )
    validate.add_argument("--candidates", required=True, type=Path)
    validate.add_argument("--concept-thresholds", required=True, type=Path)
    validate.add_argument(
        "--screening",
        nargs="+",
        required=True,
        help="One or more centre=/path/to/screening.npz inputs",
    )
    validate.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _standardize_centre_column(frame):
    if "center" in frame.columns and "centre" not in frame.columns:
        frame = frame.rename(columns={"center": "centre"})
    return frame


def _read_tables(paths):
    frames = [_standardize_centre_column(pd.read_csv(path)) for path in paths]
    return pd.concat(frames, ignore_index=True)


def run_eligibility(args):
    frame = _read_tables(args.performance)
    required = {"centre", "disease", "n_positive", "auroc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Performance tables are missing columns: {sorted(missing)}")
    if "delta_auroc" not in frame:
        if "age_sex_auroc" not in frame:
            raise ValueError("Provide either delta_auroc or age_sex_auroc")
        frame["delta_auroc"] = frame["auroc"] - frame["age_sex_auroc"]

    frame["evaluable"] = (
        (frame["n_positive"] >= 30)
        & np.isfinite(frame["auroc"])
        & np.isfinite(frame["delta_auroc"])
    )
    evaluable = frame.loc[frame["evaluable"]].copy()
    summary = evaluable.groupby("disease", as_index=False).agg(
        n_evaluable_centres=("centre", "nunique"),
        mean_auroc=("auroc", "mean"),
        mean_delta_auroc=("delta_auroc", "mean"),
    )
    summary["retained"] = (
        (summary["n_evaluable_centres"] >= 3)
        & (summary["mean_auroc"] > 0.70)
        & (summary["mean_delta_auroc"] > 0)
    )
    summary = summary.sort_values(
        ["retained", "mean_auroc", "disease"],
        ascending=[False, False, True],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Retained {int(summary['retained'].sum())} diseases in {args.output}")


def _concept_order():
    path = (
        Path(__file__).resolve().parent
        / "assets/concepts/schema/active_binary_concept_ids.json"
    )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_rank(args):
    frame = _read_tables(args.attribution)
    required = {
        "centre",
        "disease",
        "concept",
        "mean_attribution",
        "n_attributed_ecgs",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Attribution tables are missing columns: {sorted(missing)}")
    keys = ["centre", "disease", "concept"]
    if frame.duplicated(keys).any():
        raise ValueError("Attribution rows must be unique by centre, disease and concept")
    frame = frame.loc[
        (frame["n_attributed_ecgs"] > 0) & np.isfinite(frame["mean_attribution"])
    ].copy()

    percentiles = []
    for (_, _), group in frame.groupby(["centre", "concept"], sort=False):
        if len(group) < 2:
            continue
        current = group.copy()
        current["attribution_percentile"] = (
            rankdata(current["mean_attribution"], method="average") - 1
        ) / (len(current) - 1)
        percentiles.append(current)
    if not percentiles:
        raise ValueError("At least two diseases per centre and concept are required")
    centre_percentiles = pd.concat(percentiles, ignore_index=True)

    candidates = centre_percentiles.groupby(
        ["disease", "concept"], as_index=False
    ).agg(
        mean_attribution_percentile=("attribution_percentile", "mean"),
        n_attribution_centres=("centre", "nunique"),
    )
    concept_order = {name: index for index, name in enumerate(_concept_order())}
    candidates["concept_order"] = candidates["concept"].map(concept_order)
    if candidates["concept_order"].isna().any():
        unknown = candidates.loc[candidates["concept_order"].isna(), "concept"].unique()
        raise ValueError(f"Unknown active concepts: {unknown[:3].tolist()}")
    candidates = candidates.sort_values(
        ["disease", "mean_attribution_percentile", "concept_order"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    candidates["candidate_rank"] = candidates.groupby("disease").cumcount() + 1
    candidates = candidates.loc[candidates["candidate_rank"] <= args.top_k].drop(
        columns="concept_order"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    centre_percentiles.to_csv(
        args.output_dir / "centre_attribution_percentiles.csv", index=False
    )
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)
    print(
        f"Wrote {len(candidates)} top-{args.top_k} disease-concept candidates "
        f"to {args.output_dir}"
    )


def _best_mcc_threshold(target, probability):
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    if target.size == 0 or np.unique(target).size < 2:
        return math.nan, math.nan
    order = np.argsort(-probability, kind="mergesort")
    scores = probability[order]
    labels = target[order].astype(np.float64)
    true_positive = np.cumsum(labels, dtype=np.float64)
    false_positive = np.cumsum(1 - labels, dtype=np.float64)
    boundary = np.r_[scores[:-1] != scores[1:], True]
    true_positive = true_positive[boundary]
    false_positive = false_positive[boundary]
    false_negative = labels.sum() - true_positive
    true_negative = (1 - labels).sum() - false_positive
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = np.divide(
        true_positive * true_negative - false_positive * false_negative,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    finite = np.flatnonzero(np.isfinite(mcc))
    if finite.size == 0:
        return math.nan, math.nan
    best = finite[np.argmax(mcc[finite])]
    return float(scores[boundary][best]), float(mcc[best])


def run_calibrate(args):
    active = _concept_order()
    pooled = {concept: [[], []] for concept in active}
    for path in args.phenotype:
        with np.load(path, allow_pickle=False) as data:
            required = {"probabilities", "targets", "concept_ids"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            probabilities = np.asarray(data["probabilities"])
            targets = np.asarray(data["targets"])
            if probabilities.shape != targets.shape or probabilities.ndim != 2:
                raise ValueError(f"Invalid probability/target shape in {path}")
            mask = (
                np.asarray(data["mask"], dtype=bool)
                if "mask" in data.files
                else np.ones_like(targets, dtype=bool)
            )
            if mask.shape != targets.shape:
                raise ValueError(f"Invalid mask shape in {path}")
            concept_ids = [str(value) for value in data["concept_ids"].tolist()]
            if len(concept_ids) != probabilities.shape[1]:
                raise ValueError(f"Concept identifiers do not match columns in {path}")
            for column, concept in enumerate(concept_ids):
                if concept not in pooled:
                    continue
                keep = (
                    mask[:, column]
                    & np.isfinite(probabilities[:, column])
                    & np.isfinite(targets[:, column])
                )
                pooled[concept][0].append(targets[keep, column])
                pooled[concept][1].append(probabilities[keep, column])

    thresholds = {}
    for concept in active:
        target_parts, probability_parts = pooled[concept]
        if not target_parts:
            continue
        threshold, _ = _best_mcc_threshold(
            np.concatenate(target_parts),
            np.concatenate(probability_parts),
        )
        if np.isfinite(threshold):
            thresholds[concept] = threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Calibrated {len(thresholds)} active binary concepts in {args.output}")


def _benjamini_hochberg(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def _sex_values(values):
    values = np.asarray(values)
    if values.dtype.kind in "OUS":
        mapped = []
        for value in values.astype(str):
            normalized = value.strip().casefold()
            if normalized in {"male", "m", "1", "2"}:
                mapped.append(1.0)
            elif normalized in {"female", "f", "0"}:
                mapped.append(0.0)
            else:
                mapped.append(np.nan)
        return np.asarray(mapped, dtype=np.float64)
    numeric = values.astype(np.float64)
    unique = set(numeric[np.isfinite(numeric)].tolist())
    if unique <= {1.0, 2.0}:
        numeric = numeric - 1.0
    return numeric


def _fit_logistic(outcome, design, max_iterations=100, tolerance=1e-8):
    """Fit unpenalized logistic regression and return coefficients and Wald errors."""

    outcome = np.asarray(outcome, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(max_iterations):
        probability = expit(design @ coefficients)
        weight = probability * (1 - probability)
        information = design.T @ (weight[:, None] * design)
        score = design.T @ (outcome - probability)
        try:
            update = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as error:
            raise ValueError("Singular logistic-regression information matrix") from error
        coefficients = coefficients + update
        if np.max(np.abs(update)) < tolerance:
            break
    else:
        raise ValueError("Logistic regression did not converge")
    probability = expit(design @ coefficients)
    weight = probability * (1 - probability)
    information = design.T @ (weight[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as error:
        raise ValueError("Logistic-regression covariance is unavailable") from error
    standard_error = np.sqrt(np.diag(covariance))
    p_value = 2 * norm.sf(np.abs(coefficients / standard_error))
    return coefficients, standard_error, p_value


def _load_screening(specification):
    if "=" not in specification:
        raise ValueError("Screening inputs must use centre=/path/to/file.npz")
    centre, path_text = specification.split("=", 1)
    path = Path(path_text)
    data = np.load(path, allow_pickle=False)
    required = {
        "concept_probabilities",
        "disease_labels",
        "age",
        "sex",
        "concept_ids",
        "disease_names",
    }
    missing = required - set(data.files)
    if missing:
        data.close()
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return centre, path, data


def run_validate(args):
    candidates = pd.read_csv(args.candidates)
    required = {
        "disease",
        "concept",
        "mean_attribution_percentile",
        "candidate_rank",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if candidates.duplicated(["disease", "concept"]).any():
        raise ValueError("Candidate rows must be unique by disease and concept")
    with args.concept_thresholds.open(encoding="utf-8") as handle:
        thresholds = {str(key): float(value) for key, value in json.load(handle).items()}

    rows = []
    for specification in args.screening:
        centre, path, data = _load_screening(specification)
        try:
            probabilities = np.asarray(data["concept_probabilities"])
            disease_labels = np.asarray(data["disease_labels"], dtype=np.float64)
            age = np.asarray(data["age"], dtype=np.float64)
            sex = _sex_values(data["sex"])
            concept_ids = [str(value) for value in data["concept_ids"].tolist()]
            disease_names = [str(value) for value in data["disease_names"].tolist()]
            disease_mask = (
                np.asarray(data["disease_mask"], dtype=bool)
                if "disease_mask" in data.files
                else np.isfinite(disease_labels)
            )
        finally:
            data.close()
        n = len(age)
        if (
            probabilities.shape != (n, len(concept_ids))
            or disease_labels.shape != (n, len(disease_names))
            or disease_mask.shape != disease_labels.shape
            or sex.shape != (n,)
        ):
            raise ValueError(f"Array shapes are inconsistent in {path}")

        concept_lookup = {name: index for index, name in enumerate(concept_ids)}
        disease_lookup = {name: index for index, name in enumerate(disease_names)}
        finite_age = np.isfinite(age)
        age_mean = age[finite_age].mean()
        age_std = age[finite_age].std()
        age_z = (age - age_mean) / (age_std if age_std > 0 else 1.0)

        for candidate in candidates.itertuples(index=False):
            if candidate.disease not in disease_lookup or candidate.concept not in concept_lookup:
                continue
            threshold = thresholds.get(candidate.concept)
            if threshold is None:
                continue
            disease_column = disease_lookup[candidate.disease]
            concept_column = concept_lookup[candidate.concept]
            keep = (
                disease_mask[:, disease_column]
                & np.isfinite(disease_labels[:, disease_column])
                & np.isfinite(probabilities[:, concept_column])
                & np.isfinite(age_z)
                & np.isfinite(sex)
            )
            if keep.sum() < 50:
                continue
            outcome = disease_labels[keep, disease_column].astype(np.int8)
            concept_state = (
                probabilities[keep, concept_column] >= threshold
            ).astype(np.int8)
            if (
                np.unique(outcome).size < 2
                or concept_state.sum() < 10
                or (1 - concept_state).sum() < 10
                or np.sum((outcome == 1) & (concept_state == 1)) < 10
            ):
                continue
            design = np.column_stack(
                [
                    np.ones(keep.sum()),
                    concept_state,
                    age_z[keep],
                    sex[keep],
                ]
            )
            try:
                coefficients, standard_errors, p_values = _fit_logistic(
                    outcome, design
                )
            except ValueError:
                continue
            beta = float(coefficients[1])
            standard_error = float(standard_errors[1])
            p_value = float(p_values[1])
            if (
                not np.isfinite(beta)
                or not np.isfinite(standard_error)
                or not np.isfinite(p_value)
                or abs(beta) > 15
            ):
                continue
            odds_ratio = float(np.exp(beta))
            ci_low = float(np.exp(beta - 1.96 * standard_error))
            ci_high = float(np.exp(beta + 1.96 * standard_error))
            rows.append(
                {
                    "centre": centre,
                    "disease": candidate.disease,
                    "concept": candidate.concept,
                    "threshold": threshold,
                    "n": int(keep.sum()),
                    "n_disease_positive": int(outcome.sum()),
                    "n_concept_positive": int(concept_state.sum()),
                    "n_joint_positive": int(
                        np.sum((outcome == 1) & (concept_state == 1))
                    ),
                    "odds_ratio": odds_ratio,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "stable": odds_ratio <= 100 and ci_high <= 1000,
                }
            )

    evidence = pd.DataFrame(rows)
    if evidence.empty:
        raise ValueError("No support-eligible candidate associations were estimated")
    evidence["q_value"] = np.nan
    for indices in evidence.groupby(["centre", "disease"]).groups.values():
        evidence.loc[indices, "q_value"] = _benjamini_hochberg(
            evidence.loc[indices, "p_value"]
        )
    evidence["positive"] = (
        evidence["stable"]
        & (evidence["odds_ratio"] > 1)
        & (evidence["q_value"] < 0.05)
    )

    stable = evidence.loc[evidence["stable"]]
    atlas = stable.groupby(["disease", "concept"], as_index=False).agg(
        n_valid_centres=("centre", "nunique"),
        n_positive_centres=("positive", "sum"),
    )
    atlas = atlas.loc[
        (atlas["n_valid_centres"] >= 2) & (atlas["n_positive_centres"] >= 2)
    ].merge(
        candidates,
        on=["disease", "concept"],
        how="left",
        validate="one_to_one",
    )
    atlas = atlas.sort_values(
        ["disease", "candidate_rank", "concept"], kind="mergesort"
    )
    evidence = evidence.sort_values(
        ["disease", "concept", "centre"], kind="mergesort"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output_dir / "centre_evidence.csv", index=False)
    atlas.to_csv(args.output_dir / "atlas.csv", index=False)
    print(f"Wrote {len(atlas)} validated disease-concept links to {args.output_dir}")


def main():
    args = parse_args()
    if args.command == "eligibility":
        run_eligibility(args)
    elif args.command == "rank":
        run_rank(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    else:
        run_validate(args)


if __name__ == "__main__":
    main()
