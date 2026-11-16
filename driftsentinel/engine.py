"""Unsupervised feature drift, new categories, and missingness checks."""
from __future__ import annotations

import bisect
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median


MISSING = {"", "none", "null", "na", "n/a", "nan"}
EPSILON = 1e-8


def missing(value: str | None) -> bool:
    return value is None or value.strip().lower() in MISSING


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV must have unique nonempty column names")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError("CSV must have rows with consistent column counts")
    return rows


def proportions(counts: list[int], smoothing: float = EPSILON) -> list[float]:
    if not counts or any(count < 0 for count in counts):
        raise ValueError("Counts must be nonnegative and nonempty")
    total = sum(counts) + smoothing * len(counts)
    if total == 0:
        raise ValueError("Counts have zero total")
    return [(count + smoothing) / total for count in counts]


def psi(baseline: list[int], current: list[int]) -> float:
    if len(baseline) != len(current):
        raise ValueError("Distribution bin counts must match")
    a, b = proportions(baseline), proportions(current)
    return sum((q - p) * math.log(q / p) for p, q in zip(a, b))


def jensen_shannon(baseline: list[int], current: list[int]) -> float:
    if len(baseline) != len(current):
        raise ValueError("Distribution bin counts must match")
    a, b = proportions(baseline), proportions(current)
    mid = [(p + q) / 2 for p, q in zip(a, b)]
    return (sum(p * math.log2(p / m) for p, m in zip(a, mid)) +
            sum(q * math.log2(q / m) for q, m in zip(b, mid))) / 2


def quantile(values: list[float], level: float) -> float:
    ordered = sorted(values)
    position = level * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_bins(values: list[float], bins: int = 8) -> list[float]:
    if bins < 2 or not values:
        raise ValueError("Require numerical data and at least two bins")
    if len(set(values)) == 1:
        value = values[0]
        margin = max(abs(value) * 1e-9, 1e-9)
        return [value - margin, value + margin]
    return sorted(set([min(values), max(values)] +
                      [quantile(values, i / bins) for i in range(1, bins)]))


def numeric_counts(values: list[float], cuts: list[float]) -> list[int]:
    counts = [0] * (len(cuts) + 1)
    for value in values:
        counts[bisect.bisect_right(cuts, value)] += 1
    return counts


def _to_number(value: str) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def compare_column(
    baseline: list[str], current: list[str],
    name: str, psi_threshold: float = 0.2,
    missing_threshold: float = 0.1,
    bins: int = 8,
) -> dict:
    if not baseline or not current:
        raise ValueError("Require nonempty baseline and current samples")
    if psi_threshold < 0 or missing_threshold < 0:
        raise ValueError("Thresholds must be nonnegative")
    old = [value for value in baseline if not missing(value)]
    new = [value for value in current if not missing(value)]
    base_missing = 1 - len(old) / len(baseline)
    current_missing = 1 - len(new) / len(current)
    numeric = bool(old) and all(_to_number(v) is not None for v in old)
    bad_numeric = 0
    novel_rate = 0.0
    if numeric:
        old_numbers = [float(value) for value in old]
        current_numbers: list[float] = []
        for value in new:
            converted = _to_number(value)
            if converted is None:
                bad_numeric += 1
            else:
                current_numbers.append(converted)
        cuts = numeric_bins(old_numbers, bins)
        old_counts = numeric_counts(old_numbers, cuts)
        current_counts = numeric_counts(current_numbers, cuts)
        labels = [f"bin_{i}" for i in range(len(old_counts))]
        kind = "numeric"
    else:
        # Stable category vocabulary with an explicit bucket for unseen labels.
        vocabulary = sorted(set(old))
        old_counts = [old.count(category) for category in vocabulary] + [0]
        current_counter = Counter(new)
        current_counts = [current_counter[category] for category in vocabulary]
        unseen = sum(n for category, n in current_counter.items() if category not in vocabulary)
        current_counts.append(unseen)
        novel_rate = unseen / len(current) if current else 0.0
        labels = vocabulary + ["__NEW__"]
        cuts = []
        kind = "categorical"
    if sum(old_counts) == 0 or sum(current_counts) == 0:
        statistic_psi, statistic_js = None, None
    else:
        statistic_psi = psi(old_counts, current_counts)
        statistic_js = jensen_shannon(old_counts, current_counts)
    missing_change = current_missing - base_missing
    reasons = []
    if statistic_psi is not None and statistic_psi > psi_threshold:
        reasons.append("distribution_shift")
    if abs(missing_change) > missing_threshold:
        reasons.append("missingness_shift")
    if bad_numeric:
        reasons.append("invalid_numeric_values")
    if novel_rate > 0:
        reasons.append("unseen_categories")
    return {
        "name": name,
        "kind": kind,
        "baseline_samples": len(baseline),
        "current_samples": len(current),
        "baseline_missing": round(base_missing, 6),
        "current_missing": round(current_missing, 6),
        "missing_change": round(missing_change, 6),
        "invalid_numeric_values": bad_numeric,
        "unseen_category_rate": round(novel_rate, 6),
        "psi": round(statistic_psi, 6) if statistic_psi is not None else None,
        "js_divergence_bits": round(statistic_js, 6) if statistic_js is not None else None,
        "bins": labels,
        "cut_points": cuts,
        "baseline_hist": old_counts,
        "current_hist": current_counts,
        "flagged": bool(reasons),
        "reasons": reasons,
    }


def analyze(
    baseline: list[dict[str, str]], current: list[dict[str, str]],
    psi_threshold: float = 0.2, missing_threshold: float = 0.1,
) -> dict:
    if not baseline or not current:
        raise ValueError("Both datasets must have observations")
    columns = list(baseline[0])
    if any(set(row) != set(columns) for row in baseline + current):
        raise ValueError("Baseline and current schemas must match")
    comparisons = [
        compare_column(
            [row[column] for row in baseline], [row[column] for row in current],
            column, psi_threshold, missing_threshold
        )
        for column in columns
    ]
    return {
        "baseline_rows": len(baseline),
        "current_rows": len(current),
        "flagged_columns": [col["name"] for col in comparisons if col["flagged"]],
        "columns": comparisons,
        "thresholds": {"psi": psi_threshold, "missing_delta": missing_threshold},
    }
