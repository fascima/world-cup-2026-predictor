"""Validate the local team injury dataset."""

from __future__ import annotations

from pathlib import Path

from src.injury_data_validation import INJURY_DATA_PATH, REPORT_PATH, validate_injury_data


def main() -> int:
    report = validate_injury_data(INJURY_DATA_PATH, report_path=REPORT_PATH)
    print(f"Checked {Path(report['path'])}")
    print(f"Rows: {report.get('summary', {}).get('rows', 0)}")
    print(f"Usable: {report['usable']}")
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"- {error}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print(f"Wrote {REPORT_PATH}")
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
