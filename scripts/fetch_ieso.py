"""Download public IESO reports to ``data/ieso/raw/``.

Idempotent: files that already exist on disk are skipped unless
``--force`` is given. All downloads use a short timeout and a polite
retry policy. The script is intentionally the only part of the codebase
that touches the network; every other module consumes the raw CSVs /
XMLs from disk.

Data source & attribution
-------------------------
All downloaded files come from the IESO public market-reports archive at
``https://reports-public.ieso.ca/public/``. IESO publishes these reports
under section 30 of the *Electricity Act, 1998* (Ontario) and under its
own Market Rules. The data is © Independent Electricity System Operator
and is used here for academic research under the IESO Terms of Use
(https://www.ieso.ca/en/Terms-of-Use) which allow non-commercial
reproduction with attribution. Every run of this script identifies
itself via a descriptive User-Agent so the operator can trace the
request; outputs are cached locally so we never re-fetch a file we
already have.

Sources
-------
* Hourly Ontario demand (market + zonal):
  ``https://reports-public.ieso.ca/public/Demand/PUB_Demand_{YYYY}.csv``
* Hourly Ontario energy price (HOEP):
  ``https://reports-public.ieso.ca/public/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_{YYYY}.csv``
* Per-generator, hourly capability / output (monthly file):
  ``https://reports-public.ieso.ca/public/GenOutputCapabilityMonth/PUB_GenOutputCapabilityMonth_{YYYYMM}.csv``
* Fleet-level hourly output by fuel (annual XML):
  ``https://reports-public.ieso.ca/public/GenOutputbyFuelHourly/PUB_GenOutputbyFuelHourly_{YYYY}.xml``

Usage
-----
::

    python scripts/fetch_ieso.py --year 2024
    python scripts/fetch_ieso.py --year 2024 --no-monthly-gen
    python scripts/fetch_ieso.py --year 2024 --force

The default run pulls everything for the year; ``--no-monthly-gen``
skips the 12 monthly generator files (~18 MB).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://reports-public.ieso.ca/public"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "ieso" / "raw"

DEMAND_TEMPLATE = f"{BASE}/Demand/PUB_Demand_{{year}}.csv"
HOEP_TEMPLATE = f"{BASE}/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_{{year}}.csv"
FUEL_TEMPLATE = f"{BASE}/GenOutputbyFuelHourly/PUB_GenOutputbyFuelHourly_{{year}}.xml"
GEN_MONTH_TEMPLATE = f"{BASE}/GenOutputCapabilityMonth/PUB_GenOutputCapabilityMonth_{{yyyymm}}.csv"

USER_AGENT = "CapGame/0.1 (academic research prototype; contact: omar.mourssi@mail.utoronto.ca)"


def _download(url: str, dest: Path, force: bool = False, timeout: float = 60.0) -> bool:
    """Download ``url`` to ``dest``; return True if a new file was written."""
    if dest.exists() and not force:
        print(f"  [skip] {dest.name}  ({dest.stat().st_size // 1024} kB)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp.write_bytes(data)
            tmp.replace(dest)
            print(f"  [ok]   {dest.name}  ({len(data) // 1024} kB)")
            return True
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= 3:
                print(f"  [fail] {dest.name}  {exc}")
                raise
            backoff = 2.0 * attempt
            print(f"  [retry {attempt}/3 in {backoff:.0f}s] {dest.name}  {exc}")
            time.sleep(backoff)


def fetch_year(
    year: int,
    dest_dir: Path = DEFAULT_DEST,
    monthly_gen: bool = True,
    force: bool = False,
) -> list[Path]:
    """Fetch the full annual bundle for ``year``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    print(f"Fetching IESO public reports for {year} -> {dest_dir}")
    jobs: list[tuple[str, str]] = [
        (DEMAND_TEMPLATE.format(year=year), f"PUB_Demand_{year}.csv"),
        (HOEP_TEMPLATE.format(year=year), f"PUB_PriceHOEPPredispOR_{year}.csv"),
        (FUEL_TEMPLATE.format(year=year), f"PUB_GenOutputbyFuelHourly_{year}.xml"),
    ]
    if monthly_gen:
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            jobs.append(
                (
                    GEN_MONTH_TEMPLATE.format(yyyymm=yyyymm),
                    f"PUB_GenOutputCapabilityMonth_{yyyymm}.csv",
                )
            )
    for url, name in jobs:
        path = dest_dir / name
        try:
            if _download(url, path, force=force):
                written.append(path)
        except Exception as exc:
            print(f"  [warn] giving up on {name}: {exc}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download IESO public reports.")
    parser.add_argument("--year", type=int, required=True, help="Calendar year.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory (default: data/ieso/raw/).",
    )
    parser.add_argument(
        "--no-monthly-gen",
        action="store_true",
        help="Skip the 12 monthly generator capability files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files that already exist on disk.",
    )
    args = parser.parse_args(argv)
    fetch_year(
        year=args.year,
        dest_dir=args.dest,
        monthly_gen=not args.no_monthly_gen,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
