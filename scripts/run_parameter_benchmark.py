from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_sign_ar.benchmark import run_grid_search, save_chart, save_csv


def main() -> None:
    root = ROOT
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rows = run_grid_search(
        smoothing_list=(3, 5, 7),
        threshold_list=(0.50, 0.55, 0.60),
        margin_list=(0.04, 0.06, 0.08),
    )

    csv_path = out / "parameter_comparison.csv"
    png_path = out / "parameter_comparison.png"

    save_csv(rows, csv_path)
    save_chart(rows, png_path, top_k=9)

    print(f"[OK] csv -> {csv_path}")
    print(f"[OK] chart -> {png_path}")
    print(f"[Top1] {rows[0].config_id} composite={rows[0].composite_score:.2f}")


if __name__ == "__main__":
    main()
