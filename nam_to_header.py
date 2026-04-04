#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def _to_symbol_stem(nam_path: Path) -> str:
    stem = nam_path.stem.strip()
    if not stem:
        return "NamModel"

    # Split on non-alphanumeric boundaries, then form PascalCase.
    raw_parts = [p for p in re.split(r"[^A-Za-z0-9]+", stem) if p]
    if not raw_parts:
        return "NamModel"

    parts = []
    for part in raw_parts:
        if part[0].isdigit():
            part = f"M{part}"
        parts.append(part[0].upper() + part[1:])

    return "".join(parts)


def _format_float(value: float) -> str:
    # Keep enough precision for model weights while staying compact.
    return f"{format(value, '.17g')}f"


def convert_nam_to_header(nam_file: Path, header_file: Path) -> tuple[str, int]:
    with nam_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "weights" not in payload or not isinstance(payload["weights"], list):
        raise ValueError("Input .nam file must contain a 'weights' array")

    weights = payload["weights"]
    for idx, value in enumerate(weights):
        if not isinstance(value, (int, float)):
            raise ValueError(f"weights[{idx}] is not numeric")

    symbol_stem = _to_symbol_stem(nam_file)
    count_name = f"{symbol_stem}WeightsCount"
    weights_name = f"{symbol_stem}Weights"

    lines = [
        "#pragma once",
        "",
        f"static const unsigned int {count_name} = {len(weights)};",
        f"static const float {weights_name}[{count_name}] = {{",
    ]

    row = []
    for idx, value in enumerate(weights, start=1):
        row.append(_format_float(float(value)))
        if idx % 6 == 0:
            lines.append("    " + ", ".join(row) + ",")
            row = []

    if row:
        lines.append("    " + ", ".join(row) + ",")

    lines.append("};")
    lines.append("")

    header_file.parent.mkdir(parents=True, exist_ok=True)
    header_file.write_text("\n".join(lines), encoding="utf-8")

    return weights_name, len(weights)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Neural Amp Modeler .nam JSON file into a C++ header "
            "with static const weight symbols."
        )
    )
    parser.add_argument("input_nam", type=Path, help="Path to input .nam file")
    parser.add_argument("output_header", type=Path, help="Path to output .h file")
    args = parser.parse_args()

    if not args.input_nam.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_nam}")

    weights_name, count = convert_nam_to_header(args.input_nam, args.output_header)
    print(f"Generated {args.output_header} with {count} weights in symbol {weights_name}.")


if __name__ == "__main__":
    main()
