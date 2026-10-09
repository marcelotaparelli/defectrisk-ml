"""Rank CSV modules using the existing verified, calibrated RF artifact."""

import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

from defectrisk.final_artifact import DESTINATION, load_artifact, validate_features
from defectrisk.model_spec import FEATURES

IDENTIFIERS = ('module', 'module_id', 'id')
MISSING_VALUES = {'', 'na', 'n/a', 'nan'}
RISK_NOTE = 'Scores are estimated defect risk, not certainty; use them to prioritize human review.'


def read_modules(path, *, identifier_column=None):
    """Keep feature order exact; never pass identifiers to the model."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError('Input CSV is empty.')
        if len(set(header)) != len(header):
            raise ValueError('CSV column names must be unique.')
        if identifier_column is not None:
            if identifier_column in FEATURES or identifier_column == 'defects':
                raise ValueError('An identifier must not be a model feature or the target.')
            if identifier_column not in header:
                raise ValueError(f'Identifier column {identifier_column!r} is missing.')
            identifier = identifier_column
        else:
            present = [name for name in IDENTIFIERS if name in header]
            if len(present) > 1:
                raise ValueError('Only one optional identifier column is allowed.')
            identifier = present[0] if present else None
        model_columns = [name for name in header if name != identifier]
        if tuple(model_columns) != FEATURES:
            raise ValueError('CSV must contain exactly the 21 original JM1 features in schema order, '
                             'plus at most one identifier (module, module_id, id, or --id-column NAME).')
        rows = []
        for number, row in enumerate(reader, 2):
            if not row:  # Ignore blank CSV lines, not rows containing missing values.
                continue
            if len(row) != len(header):
                raise ValueError(f'CSV record at line {number} must have {len(header)} fields.')
            rows.append(row)
    if not rows:
        raise ValueError('CSV contains no modules.')
    frame = pd.DataFrame(rows, columns=header)
    if identifier is None:
        modules = [f'row_{number}' for number in range(1, len(frame) + 1)]
    else:
        modules = frame[identifier].tolist()
        if any(not name.strip() for name in modules):
            raise ValueError('Supplied module identifiers must not be empty.')
    features = pd.DataFrame(index=frame.index)
    for name in FEATURES:
        values = frame[name].str.strip()
        values = values.mask(values.str.lower().isin(MISSING_VALUES), float('nan'))
        try:
            features[name] = pd.to_numeric(values, errors='raise').astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f'Feature {name!r} must contain numeric values or missing values.') from error
    validate_features(features)
    return modules, features


def rank_modules(path, *, artifact=DESTINATION, identifier_column=None):
    modules, features = read_modules(path, identifier_column=identifier_column)
    try:
        model = load_artifact(artifact)
    except Exception as error:
        raise ValueError(f'Cannot load verified artifact at {artifact}: {error}') from error
    predictions = model.predict(features)
    # Stable ordering preserves CSV order when probabilities are exactly tied.
    ranked = sorted(zip(modules, predictions), key=lambda row: -row[1].probability)
    return [
        {'rank': number, 'module': module, 'risk_probability': prediction.probability,
         'model_version': prediction.model_version, 'calibration': prediction.calibration_method}
        for number, (module, prediction) in enumerate(ranked, 1)
    ]


def render_table(rows):
    # Escape control characters in identifiers so each module occupies one line.
    names = [json.dumps(row['module'], ensure_ascii=False)[1:-1] for row in rows]
    rank_width = max(4, len(str(len(rows))))
    module_width = max(6, max(map(len, names)))
    lines = [f'{"RANK":<{rank_width}}  {"MODULE":<{module_width}}  RISK']
    lines.extend(f'{row["rank"]:<{rank_width}}  {name:<{module_width}}  {row["risk_probability"]:.4f}'
                 for row, name in zip(rows, names))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='defectrisk', description=__doc__, epilog=RISK_NOTE)
    commands = parser.add_subparsers(dest='command', required=True)
    rank = commands.add_parser('rank', help='Rank modules by estimated calibrated defect risk.', epilog=RISK_NOTE)
    rank.add_argument('csv', type=Path, help='CSV with the ordered original 21 JM1 features.')
    rank.add_argument('--artifact', type=Path, default=DESTINATION, help='Verified artifact directory (default: %(default)s).')
    rank.add_argument('--format', choices=['table', 'json'], default='table')
    rank.add_argument('--id-column', help='One custom identifier column; never passed into the model.')
    args = parser.parse_args(argv)
    try:
        rows = rank_modules(args.csv, artifact=args.artifact, identifier_column=args.id_column)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        parser.error(str(error))
    print(RISK_NOTE, file=sys.stderr)
    if args.format == 'json':
        print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(render_table(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
