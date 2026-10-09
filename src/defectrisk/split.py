"""Create and report the M3 holdout split without fitting anything."""

import numpy as np
import pandas as pd

from defectrisk.data import TARGET, load_jm1


def feature_groups(X: pd.DataFrame) -> pd.Series:
    """Group by every raw feature, treating matching missing values as equal."""
    if TARGET in X.columns:
        raise ValueError(f"Target {TARGET!r} must not be included in features.")
    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    return X.groupby(list(X.columns), dropna=False, sort=False, observed=True).ngroup()


def group_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Allocate whole feature groups to one deterministic, balanced holdout.

    Largest groups are considered first; seed 42 breaks size ties. Greedily
    include a group when it reduces squared deviations from the desired
    held-out row count and per-class counts, normalized by their full totals.
    Then move or exchange whole groups while that allocation cost improves.
    Labels inform allocation only, never group identity. This is one holdout,
    not cross-validation or model selection. Indivisible groups can prevent
    exact sizes or class balance. Preserve original row order and indices.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a fraction strictly between 0 and 1.")
    if not X.index.is_unique:
        raise ValueError("Dataset must have unique row index labels.")
    if not X.index.equals(y.index):
        raise ValueError("Features and labels must have matching indices.")
    if y.isna().any():
        raise ValueError("Target labels must not be missing.")
    groups = feature_groups(X)
    if groups.nunique() < 2:
        raise ValueError("At least two distinct feature groups are required for a holdout.")
    counts = pd.crosstab(groups.to_numpy(), y.to_numpy()).to_numpy(dtype=float)
    # Include total row count as well as each class count in the allocation cost.
    counts = np.column_stack([counts.sum(axis=1), counts])
    totals = counts.sum(axis=0)
    desired = totals * test_size
    order = np.random.RandomState(random_state).permutation(len(counts))
    order = order[np.argsort(-counts[order, 0], kind="stable")]
    selected = np.zeros(len(counts), dtype=bool)
    held_out = np.zeros(counts.shape[1])

    def cost(values):
        return np.square((values - desired) / totals).sum()

    for group in order:
        if cost(held_out + counts[group]) < cost(held_out):
            selected[group] = True
            held_out += counts[group]
    # Small groups can correct deviations introduced by early large groups.
    # This balances partition counts only; no model or metric is consulted.
    changed = True
    while changed:
        changed = False
        for group in order[::-1]:
            candidate = held_out + (-counts[group] if selected[group] else counts[group])
            if cost(candidate) < cost(held_out) - 1e-15:
                selected[group] = not selected[group]
                held_out = candidate
                changed = True
        if not changed:
            # An exchange can improve balance even when neither individual move
            # helps. Groups with the same class-count profile are interchangeable
            # for this cost; consider one deterministic representative per side.
            representatives = [{}, {}]
            for group in order[::-1]:
                representatives[int(selected[group])].setdefault(tuple(counts[group]), group)
            best_cost = cost(held_out) - 1e-15
            exchange = None
            for outgoing in representatives[1].values():
                for incoming in representatives[0].values():
                    candidate = held_out - counts[outgoing] + counts[incoming]
                    candidate_cost = cost(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        exchange = outgoing, incoming, candidate
            if exchange is not None:
                outgoing, incoming, held_out = exchange
                selected[outgoing], selected[incoming] = False, True
                changed = True
    # Very large groups may make the requested ratio impossible. Still return
    # two nonempty partitions without ever breaking a group.
    if not selected.any():
        selected[min(order, key=lambda group: cost(counts[group]))] = True
    elif selected.all():
        selected[min(order, key=lambda group: cost(held_out - counts[group]))] = False
    test_mask = selected[groups.to_numpy(dtype=int)]
    train_mask = ~test_mask
    if set(groups[train_mask]) & set(groups[test_mask]):
        raise RuntimeError("Feature groups overlap across the holdout boundary.")
    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask]


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test with a group-aware holdout.

    Keep original row indices and all feature values, including missing values
    and duplicate content. Unique index labels are required for row tracking.
    The defaults define the project's fixed M3 split.
    """
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    if not frame.index.is_unique:
        raise ValueError("Dataset must have unique row index labels.")

    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    return group_holdout(X, y, test_size=test_size, random_state=random_state)


def split_report(frame: pd.DataFrame) -> str:
    """Report sizes and class balance for the fixed split; make no decisions."""
    X_train, X_test, y_train, y_test = split_dataset(frame)
    groups = feature_groups(frame.drop(columns=TARGET))
    overlap = len(set(groups.loc[X_train.index]) & set(groups.loc[X_test.index]))
    lines = [
        "Train/Test Split — group-aware, test_size=0.20, random_state=42",
        "Groups: all raw feature columns, excluding target; approximate class balance",
        f"Full dataset size: {len(frame):,}",
        f"Train size: {len(X_train):,}",
        f"Test size: {len(X_test):,}",
        f"Shared feature vectors (train/test): {overlap}",
    ]
    for name, labels in [("Full", frame[TARGET]), ("Train", y_train), ("Test", y_test)]:
        counts = labels.value_counts().sort_index()
        distribution = pd.DataFrame(
            {"count": counts, "percent": counts / len(labels) * 100}
        )
        lines.extend(
            [
                f"\n{name} class distribution:",
                distribution.to_string(float_format=lambda value: f"{value:.2f}"),
            ]
        )
    return "\n".join(lines)


def main() -> None:
    print(split_report(load_jm1()))


if __name__ == "__main__":
    main()
