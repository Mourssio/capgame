"""Run the Ontario 2024 -> 2050 forecast and dump tidy CSVs.

Usage
-----
::

    python scripts/forecast_ontario.py --out reports/forecast
    python scripts/forecast_ontario.py --out reports/forecast --monte-carlo 200

Outputs written to ``out/``:

* ``fleet.csv``  -- nameplate MW by technology by year
* ``trajectory_deterministic.csv`` -- year-indexed equilibrium under
  energy-only oligopoly on the deterministic pathway
* ``mechanism_matrix_trajectory.csv`` -- per-year mechanism x
  structure matrix
* ``monte_carlo_paths.csv`` -- (if requested) all simulated paths
* ``monte_carlo_bands.csv`` -- (if requested) P10/P50/P90 bands
* ``summary.txt`` -- one-page human-readable report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from capgame.calibration.ontario import DEFAULT_RAW_DIR, build_ontario_scenario
from capgame.forecast import (
    MonteCarloConfig,
    build_trajectory,
    default_ontario_pathway,
    run_mechanism_matrix_trajectory,
    run_monte_carlo,
    run_trajectory,
    summarize_paths,
)


def _write_fleet(pathway, years, out: Path) -> pd.DataFrame:
    rows = []
    for y in years:
        row = {"year": y}
        row.update(pathway.fleet_mw_at(y))
        row["peak_mw"] = pathway.peak_mw_at(y)
        row["mean_mw"] = pathway.mean_mw_at(y)
        row["gas_price_per_mmbtu"] = pathway.gas_price.price_at(y)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fleet.csv", index=False)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024, help="Base-year calibration.")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--horizon", type=int, default=2050)
    ap.add_argument("--out", type=Path, default=Path("reports/forecast"))
    ap.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        metavar="N_PATHS",
        help="Run N_PATHS Monte Carlo trajectories (0 = skip).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Calibrating base year {args.year} from {args.raw_dir}...")
    cal = build_ontario_scenario(year=args.year, raw_dir=args.raw_dir)
    pathway = default_ontario_pathway()
    years = list(range(args.year, args.horizon + 1))

    print(f"Writing fleet trajectory ({years[0]} -> {years[-1]})...")
    _write_fleet(pathway, years, args.out)

    print("Running deterministic trajectory (energy-only, oligopoly)...")
    traj = build_trajectory(cal, pathway, years)
    df_det = run_trajectory(traj, include_per_firm=True)
    df_det.to_csv(args.out / "trajectory_deterministic.csv", index=False)
    if "per_firm" in df_det.attrs:
        df_det.attrs["per_firm"].to_csv(args.out / "trajectory_per_firm.csv", index=False)

    print("Running mechanism x structure matrix at every year...")
    df_matrix = run_mechanism_matrix_trajectory(traj, structures=("oligopoly", "cartel"))
    df_matrix.to_csv(args.out / "mechanism_matrix_trajectory.csv", index=False)

    if args.monte_carlo > 0:
        print(f"Running Monte Carlo ({args.monte_carlo} paths)...")
        mc = run_monte_carlo(
            cal,
            years=years,
            config=MonteCarloConfig(n_paths=args.monte_carlo, seed=args.seed),
        )
        mc.to_csv(args.out / "monte_carlo_paths.csv", index=False)
        bands = summarize_paths(mc)
        bands.to_csv(args.out / "monte_carlo_bands.csv", index=False)

    # One-page summary text.
    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Ontario forecast {years[0]} -> {years[-1]} (base pathway: {pathway.name})\n")
        f.write("=" * 72 + "\n\n")
        snap = df_det.set_index("year")
        f.write("Deterministic pathway highlights (energy-only, oligopoly)\n")
        f.write("-" * 72 + "\n")
        for y in (years[0], 2030, 2035, 2040, 2050):
            if y not in snap.index:
                continue
            r = snap.loc[y]
            f.write(
                f"  {y}:  price={r['expected_price']:>6.2f} $/MWh   "
                f"Q={r['expected_quantity_mw']:>8,.0f} MW   "
                f"missing_money={r['fleet_missing_money_per_year']:>12,.0f} $/yr   "
                f"reserve={r['reserve_margin']:>5.2f}\n"
            )
        if args.monte_carlo > 0:
            f.write("\nMonte Carlo P10/P50/P90 of fleet missing money ($/yr)\n")
            f.write("-" * 72 + "\n")
            bands_s = bands.set_index("year")
            for y in (2030, 2035, 2040, 2050):
                if y not in bands_s.index:
                    continue
                f.write(
                    f"  {y}:  "
                    f"P10={bands_s.loc[y, 'fleet_missing_money_per_year_q10']:>12,.0f}   "
                    f"P50={bands_s.loc[y, 'fleet_missing_money_per_year_q50']:>12,.0f}   "
                    f"P90={bands_s.loc[y, 'fleet_missing_money_per_year_q90']:>12,.0f}\n"
                )

    print(f"\nAll outputs written to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
