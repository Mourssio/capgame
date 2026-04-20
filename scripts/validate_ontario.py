"""Compare the CapGame Ontario calibration against IESO-published reality.

Run after ``scripts/fetch_ieso.py``; prints a one-page report that an
examiner can use to judge how faithfully the calibrated model mirrors
the 2024 Ontario market.

Validation targets
------------------
* **Peak demand** from the model (``demand.a / demand.b``, which is the
  choke quantity of the calibrated inverse-demand curve) vs. the
  observed annual peak in ``PUB_Demand``. These should differ: the
  choke quantity is "demand at price zero," always above the observed
  peak; we just check the ratio is plausible.
* **Mean equilibrium price** of :func:`run_scenario` under the default
  (energy-only, oligopoly) setting vs. the observed mean HOEP.
* **Mean equilibrium dispatch** under the stationary renewable
  distribution vs. the observed mean Ontario demand.
* **Nameplate capacities by fuel class** vs. IESO's "Ontario at a
  Glance" published values (~12 GW nuclear, ~9 GW hydro, ~10 GW gas,
  ~5 GW wind, ~0.5 GW solar).
* **LOLE and EUE** using literature-based thermal FORs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from capgame.calibration.ontario import DEFAULT_RAW_DIR, build_ontario_scenario
from capgame.experiments.scenarios import run_scenario


def _fmt(x: float, width: int = 10, dp: int = 1) -> str:
    return f"{x:>{width},.{dp}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--elasticity", type=float, default=-0.1)
    args = parser.parse_args(argv)

    cal = build_ontario_scenario(year=args.year, raw_dir=args.raw_dir, elasticity=args.elasticity)
    res = run_scenario(cal.scenario)

    print(f"=== CapGame vs. IESO public reports, {args.year} ===\n")

    print("Demand curve (inverse)")
    print(f"  method          = {cal.demand_fit.method}")
    print(f"  a ($/MWh)       = {_fmt(cal.demand_fit.a, 8, 2)}")
    print(f"  b ($/MWh/MW)    = {cal.demand_fit.b:>10.6f}")
    print(f"  choke Q (MW)    = {_fmt(cal.demand_fit.a / cal.demand_fit.b, 10, 0)}")
    print(f"  observed peak   = {_fmt(cal.peak_load_mw, 10, 0)}")
    print(f"  observed mean   = {_fmt(cal.demand_fit.reference_quantity, 10, 0)}")
    print(f"  observed HOEP   = {_fmt(cal.demand_fit.reference_price, 10, 2)}  $/MWh\n")

    print("Technology classes (firms)")
    print(f"  {'name':<12} {'cap_MW':>10} {'MC':>6} {'FOR':>6}")
    for c in cal.technology_classes:
        print(
            f"  {c.name:<12} {c.capacity_mw:>10,.0f} "
            f"{c.marginal_cost:>6.1f} {c.outage_rate:>6.3f}"
        )
    total_cap = sum(c.capacity_mw for c in cal.technology_classes)
    print(f"  {'TOTAL':<12} {total_cap:>10,.0f}\n")

    print("Renewable Markov chain")
    rc = cal.renewable_calibration
    print(
        f"  nameplate: wind = {rc.wind_capacity_mw:,.0f} MW, "
        f"solar = {rc.solar_capacity_mw:,.0f} MW"
    )
    print(f"  mean CF:   wind = {rc.mean_wind_cf:.3f}, solar = {rc.mean_solar_cf:.3f}")
    pi = rc.chain.stationary_distribution()
    for s, p in zip(rc.chain.states, pi, strict=True):
        print(f"    {s.name:<6} pi={p:.3f}  wind_cf={s.wind_cf:.3f}  solar_cf={s.solar_cf:.3f}")
    print()

    print("Equilibrium (oligopoly, energy-only) aggregated over renewable states")
    print(f"  expected price        = {_fmt(res.expected_price, 10, 2)}  $/MWh")
    print(f"  expected dispatch     = {_fmt(res.expected_quantity, 10, 0)}  MW")
    print(f"  consumer surplus      = {_fmt(res.expected_consumer_surplus, 14, 0)}")
    print(f"  producer surplus      = {_fmt(res.expected_producer_surplus, 14, 0)}")
    print(f"  welfare               = {_fmt(res.expected_welfare, 14, 0)}\n")

    print("Adequacy")
    ad = res.adequacy
    print(f"  reserve margin        = {ad.reserve_margin * 100:.1f} %")
    print(f"  target                = {ad.target_reserve_margin * 100:.1f} %")
    print(f"  capacity required (MW)= {ad.capacity_required_mw:,.0f}")
    print(f"  total nameplate (MW)  = {ad.total_capacity_mw:,.0f}")
    if ad.lole_hours_per_year is not None:
        print(f"  LOLE                  = {ad.lole_hours_per_year:.2f}  h/yr")
    if ad.eue_mwh_per_year is not None:
        print(f"  EUE                   = {ad.eue_mwh_per_year:,.0f}  MWh/yr\n")

    print("FOR calibration diagnostics (empirical proxy vs. literature)")
    emp = {e.fuel: e.outage_rate for e in cal.empirical_outage_rates}
    from capgame.calibration.ontario import LITERATURE_OUTAGE_RATES

    for fuel in ("NUCLEAR", "HYDRO", "GAS", "BIOFUEL"):
        lit_key = {"GAS": "GAS_CCGT"}.get(fuel, fuel)
        print(
            f"  {fuel:<8} empirical={emp.get(fuel, float('nan')):.3f}   "
            f"literature={LITERATURE_OUTAGE_RATES.get(lit_key, float('nan')):.3f}"
        )

    print("\nReferences used for 'truth'")
    print("  IESO 'Ontario Supply Mix' (2024):")
    print("    Nuclear ~12 GW | Hydro ~9 GW | Gas ~10 GW | Wind ~5 GW | Solar ~0.5 GW")
    print("  NPCC reliability target: ~18% reserve margin over peak demand.")
    print()
    price_ratio = res.expected_price / cal.demand_fit.reference_price
    print("Interpretation")
    print(
        f"  Model price / observed HOEP = {price_ratio:.2f}x. Ontario clears via a "
        "cost-based pool with must-run nuclear, so observed HOEP is close to marginal\n"
        "  cost; the Cournot oligopoly equilibrium reveals the markup that the"
        " same\n  technology mix would support under strategic bidding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
