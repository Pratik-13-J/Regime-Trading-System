"""
fetch_stocks.py — Fetch Historical Data for Stock Universe
===========================================================
Run this once to download NIFTY + all stocks in your universe.
Saves clean CSVs to data/raw/ for all downstream modules.

Usage:
    python fetch_stocks.py

Add/remove stocks from STOCK_UNIVERSE dict below.
Tokens sourced from Angel One scrip master.

How to find tokens manually (if auto-lookup fails):
    https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
    Search for your symbol + "-EQ" in the NSE segment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data.fetcher import AngelOneFetcher, lookup_token
import logging

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# YOUR STOCK UNIVERSE — Edit this list
# Format: "SYMBOL" : "ANGEL_ONE_TOKEN"
#
# To find tokens automatically, set token to None
# and the script will look them up from Angel One scrip master.
# ═══════════════════════════════════════════════════════════════════

STOCK_UNIVERSE = {
    # Large Cap — NIFTY 50 constituents
    "RELIANCE"   : "2885",
    "INFY"       : "1594",
    "TCS"        : "11536",
    "HDFCBANK"   : "1333",
    "ICICIBANK"  : "4963",
    "SBIN"       : "3045",
    "WIPRO"      : "3787",
    "AXISBANK"   : "5900",
    "KOTAKBANK"  : "1922",
}


# ═══════════════════════════════════════════════════════════════════
# AUTO TOKEN LOOKUP (for any symbol where token = None)
# ═══════════════════════════════════════════════════════════════════

def resolve_tokens(universe: dict) -> dict:
    """
    For any symbol with token = None, attempt auto-lookup
    from Angel One scrip master JSON.
    Removes symbols where token could not be found.
    """
    resolved = {}
    failed   = []

    for symbol, token in universe.items():
        if token is not None:
            resolved[symbol] = token
        else:
            print(f"  Looking up token for {symbol}...")
            found = lookup_token(symbol)
            if found:
                resolved[symbol] = found
                print(f"  ✅ {symbol} → {found}")
            else:
                failed.append(symbol)
                print(f"  ❌ {symbol} → token not found, skipping")

    if failed:
        print(f"\n⚠️  Could not find tokens for: {failed}")
        print("   Add them manually to STOCK_UNIVERSE dict.")

    return resolved


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  Regime Trading System — Stock Universe Data Fetch")
    print("=" * 60)
    print(f"\n  Stocks to fetch : {len(STOCK_UNIVERSE)}")
    print(f"  Symbols         : {list(STOCK_UNIVERSE.keys())}\n")

    # Step 1 — Resolve any missing tokens
    stocks = resolve_tokens(STOCK_UNIVERSE)
    print(f"\n✅ Tokens resolved for {len(stocks)} stocks\n")

    # Step 2 — Initialize fetcher (handles login)
    fetcher = AngelOneFetcher()

    # Step 3 — Fetch NIFTY + all stocks
    # fetch_all() always fetches NIFTY first, then each stock
    results = fetcher.fetch_all(stocks=stocks)

    # Step 4 — Summary
    print("\n" + "=" * 60)
    print("  FETCH COMPLETE — Summary")
    print("=" * 60)
    print(f"  {'Symbol':<15} {'Rows':>8}  {'From':<12}  {'To':<12}")
    print(f"  {'-'*50}")

    for symbol, df in results.items():
        print(
            f"  {symbol:<15} "
            f"{len(df):>8,}  "
            f"{str(df['Datetime'].min().date()):<12}  "
            f"{str(df['Datetime'].max().date()):<12}"
        )

    print(f"\n  All CSVs saved to: data/raw/")
    print(f"  Total symbols    : {len(results)}")
    print("=" * 60)