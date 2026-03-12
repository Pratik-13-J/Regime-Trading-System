"""
fetcher.py — Angel One SmartAPI Data Fetcher
=============================================
Fetches historical OHLCV data for NIFTY and any stock symbol.
Handles Angel One API limitations cleanly:
    - Max ~500 records per request → fetches in date chunks
    - Stitches chunks → deduplicates → sorts → validates
    - Retries on transient failures
    - Saves clean CSVs consumed by all downstream modules

Usage:
    from src.data.fetcher import AngelOneFetcher

    fetcher = AngelOneFetcher()
    fetcher.fetch_nifty()                        # saves nifty50_ohlc_daily.csv
    fetcher.fetch_stock("RELIANCE", "2885")      # saves RELIANCE_ohlc_daily.csv
    fetcher.fetch_all(stocks={"RELIANCE":"2885"})# fetches both in one call
"""

import time
import logging
import pyotp
import pandas as pd
import numpy as np

from SmartApi import SmartConnect
from datetime import datetime, timedelta
from typing import Optional, Dict

from config import CONFIG

# ── Logger ──────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Angel One API returns max 500 records per request.
# For daily data: 400 days per chunk gives safe margin.
CHUNK_DAYS      = 400

# Seconds to wait between API calls (avoid rate limiting)
API_DELAY_SEC   = 0.5

# Max retries on transient API failure
MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2

# Angel One date format requirement — MUST be exactly this format
ANGEL_DATE_FMT  = "%Y-%m-%d %H:%M"

# Standard columns returned by getCandleData
OHLCV_COLUMNS   = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


# ═══════════════════════════════════════════════════════════════════
# FETCHER CLASS
# ═══════════════════════════════════════════════════════════════════

class AngelOneFetcher:
    """
    Handles login, chunked data fetching, validation, and CSV saving
    for Angel One SmartAPI.

    Typical flow:
        fetcher = AngelOneFetcher()        # authenticates on init
        fetcher.fetch_nifty()              # NIFTY 50 daily data
        fetcher.fetch_stock("RELIANCE", "2885")  # stock daily data
    """

    def __init__(self):
        """
        Authenticate with Angel One SmartAPI using TOTP.
        Credentials loaded from environment variables via CONFIG.
        Raises EnvironmentError if credentials missing.
        Raises ConnectionError if login fails.
        """
        CONFIG.angel.validate()    # fail loud if credentials missing
        self._obj    = None
        self._logged_in = False
        self._login()

    # ──────────────────────────────────────────────────────────────
    # AUTHENTICATION
    # ──────────────────────────────────────────────────────────────

    def _login(self):
        """
        Login to Angel One using TOTP-based 2FA (pyotp).
        Stores authenticated SmartConnect object for all subsequent calls.
        """
        log.info("Authenticating with Angel One SmartAPI...")

        try:
            self._obj = SmartConnect(api_key=CONFIG.angel.api_key)

            # Generate live TOTP token
            totp_token = pyotp.TOTP(CONFIG.angel.totp_secret).now()

            session = self._obj.generateSession(
                clientCode = CONFIG.angel.client_id,
                password   = CONFIG.angel.password,
                totp       = totp_token,
            )

            if not session.get("status", False):
                raise ConnectionError(
                    f"Angel One login failed. Response: {session}"
                )

            self._logged_in = True
            log.info(f"✅ Login successful | Client: {CONFIG.angel.client_id}")

        except Exception as e:
            raise ConnectionError(f"Angel One authentication failed: {e}") from e

    # ──────────────────────────────────────────────────────────────
    # DATE RANGE CHUNKING
    # ──────────────────────────────────────────────────────────────

    def _build_date_chunks(
        self,
        start_date : str,
        end_date   : str,
        chunk_days : int = CHUNK_DAYS,
    ) -> list[tuple[str, str]]:
        """
        Split a long date range into chunks of chunk_days each.
        Angel One caps at ~500 records per request — chunking avoids
        silent data truncation.

        Returns list of (from_str, to_str) tuples in Angel One format:
            "YYYY-MM-DD HH:MM"

        Example:
            start = "2015-01-01",  end = "2024-12-31", chunk = 400 days
            → [("2015-01-01 09:00", "2016-02-05 15:30"),
               ("2016-02-06 09:00", "2017-03-12 15:30"), ...]
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")

        chunks = []
        current = start

        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((
                current.strftime(ANGEL_DATE_FMT).replace("00:00", "09:00"),
                chunk_end.strftime(ANGEL_DATE_FMT).replace("00:00", "15:30"),
            ))
            current = chunk_end + timedelta(days=1)

        log.info(f"Date range split into {len(chunks)} chunks of ~{chunk_days} days each.")
        return chunks

    # ──────────────────────────────────────────────────────────────
    # SINGLE CHUNK FETCH (with retry)
    # ──────────────────────────────────────────────────────────────

    def _fetch_chunk(
        self,
        symbol_token : str,
        exchange     : str,
        interval     : str,
        from_date    : str,
        to_date      : str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch one date-range chunk from Angel One API with retry logic.
        Returns DataFrame or None if chunk returns no data (e.g. holiday period).
        """
        params = {
            "exchange"   : exchange,
            "symboltoken": symbol_token,
            "interval"   : interval,
            "fromdate"   : from_date,
            "todate"     : to_date,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._obj.getCandleData(params)

                if not response.get("status", False):
                    error = response.get("message", "Unknown error")
                    log.warning(f"  Chunk {from_date}→{to_date} | API error: {error} (attempt {attempt})")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_SEC)
                    continue

                data = response.get("data", [])

                if not data:
                    # Empty chunk — likely a holiday-only period, skip silently
                    return None

                df = pd.DataFrame(data, columns=OHLCV_COLUMNS)
                return df

            except Exception as e:
                log.warning(f"  Chunk {from_date}→{to_date} | Exception: {e} (attempt {attempt})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)

        log.error(f"  Chunk {from_date}→{to_date} | Failed after {MAX_RETRIES} attempts. Skipping.")
        return None

    # ──────────────────────────────────────────────────────────────
    # FULL FETCH (all chunks stitched)
    # ──────────────────────────────────────────────────────────────

    def _fetch_full(
        self,
        symbol_token : str,
        exchange     : str,
        interval     : str,
        start_date   : str,
        end_date     : str,
        label        : str = "symbol",
    ) -> pd.DataFrame:
        """
        Fetch complete historical data by looping over date chunks.
        Stitches all chunks, deduplicates, and sorts by Datetime.

        Args:
            symbol_token : Angel One token string (e.g. "99926000")
            exchange     : "NSE" or "BSE"
            interval     : "ONE_DAY", "ONE_HOUR", etc.
            start_date   : "YYYY-MM-DD"
            end_date     : "YYYY-MM-DD"
            label        : Human-readable name for logging

        Returns:
            Clean DataFrame with columns: Datetime, Open, High, Low, Close, Volume
        """
        end = end_date or datetime.today().strftime("%Y-%m-%d")
        chunks = self._build_date_chunks(start_date, end)

        all_frames = []
        log.info(f"Fetching {label} | {start_date} → {end} | {len(chunks)} chunks...")

        for i, (from_dt, to_dt) in enumerate(chunks, 1):
            log.info(f"  [{i:>3}/{len(chunks)}] {from_dt} → {to_dt}")
            df_chunk = self._fetch_chunk(symbol_token, exchange, interval, from_dt, to_dt)

            if df_chunk is not None and len(df_chunk) > 0:
                all_frames.append(df_chunk)

            time.sleep(API_DELAY_SEC)    # respect rate limits

        if not all_frames:
            raise ValueError(
                f"No data returned for {label} "
                f"({start_date} → {end}). "
                f"Check symbol token and date range."
            )

        # Stitch all chunks
        df = pd.concat(all_frames, ignore_index=True)

        # Clean and standardize
        df = self._clean(df, label)

        log.info(
            f"✅ {label} | {len(df)} rows | "
            f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}"
        )
        return df

    # ──────────────────────────────────────────────────────────────
    # DATA CLEANING & VALIDATION
    # ──────────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Clean raw API response:
            1. Parse Datetime (handles Angel One's IST timezone string)
            2. Remove duplicates (chunk overlap)
            3. Sort by date ascending
            4. Drop rows with NaN in OHLC
            5. Enforce correct dtypes
            6. Validate OHLC integrity (High >= Low, etc.)
        """
        # 1. Parse Datetime — Angel One returns ISO8601 with timezone
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_localize(None)
        # For daily data: keep date only (strip time component)
        df["Datetime"] = df["Datetime"].dt.normalize()

        # 2. Deduplicate (chunk boundaries may overlap by 1 row)
        df = df.drop_duplicates(subset=["Datetime"]).reset_index(drop=True)

        # 3. Sort ascending
        df = df.sort_values("Datetime").reset_index(drop=True)

        # 4. Drop NaN OHLC rows
        ohlc_cols = ["Open", "High", "Low", "Close"]
        before = len(df)
        df = df.dropna(subset=ohlc_cols).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            log.warning(f"  {label}: Dropped {dropped} rows with NaN OHLC values.")

        # 5. Enforce dtypes
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

        # 6. OHLC integrity checks
        bad_hl = (df["High"] < df["Low"]).sum()
        bad_co = (df["Close"] <= 0).sum()
        if bad_hl > 0:
            log.warning(f"  {label}: {bad_hl} rows where High < Low. Dropping.")
            df = df[df["High"] >= df["Low"]].reset_index(drop=True)
        if bad_co > 0:
            log.warning(f"  {label}: {bad_co} rows where Close <= 0. Dropping.")
            df = df[df["Close"] > 0].reset_index(drop=True)

        return df

    # ──────────────────────────────────────────────────────────────
    # VALIDATION & PREVIEW
    # ──────────────────────────────────────────────────────────────

    def _validate_and_preview(self, df: pd.DataFrame, label: str):
        """
        Post-fetch validation and data preview printed to console.
        Checks minimum row requirement from CONFIG.
        Prints a clean summary table.
        """
        print(f"\n{'═'*55}")
        print(f"  📊 DATA PREVIEW — {label}")
        print(f"{'═'*55}")
        print(f"  Rows          : {len(df):,}")
        print(f"  Date range    : {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
        print(f"  Trading days  : {len(df):,}")

        # Check for date gaps > 5 business days (potential missing data)
        date_diffs = df["Datetime"].diff().dt.days.dropna()
        large_gaps = date_diffs[date_diffs > 7]
        if len(large_gaps) > 0:
            log.warning(f"  {label}: {len(large_gaps)} gaps > 7 days detected (likely holidays/halts).")

        print(f"\n  OHLCV Stats:")
        print(f"  {'Metric':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}")
        print(f"  {'-'*52}")
        for stat, row in df[["Open","High","Low","Close"]].describe().loc[["min","mean","max"]].iterrows():
            print(f"  {stat:<12} {row['Open']:>10.2f} {row['High']:>10.2f} {row['Low']:>10.2f} {row['Close']:>10.2f}")

        print(f"\n  Last 3 rows:")
        print(df[["Datetime","Open","High","Low","Close","Volume"]].tail(3).to_string(index=False))

        # Minimum rows check
        min_required = CONFIG.data.min_rows_required
        if len(df) < min_required:
            raise ValueError(
                f"{label} has only {len(df)} rows — minimum required is {min_required}. "
                f"Extend start_date in CONFIG.data.start_date."
            )
        print(f"\n  ✅ Minimum rows check passed ({len(df)} >= {min_required})")
        print(f"{'═'*55}\n")

    # ──────────────────────────────────────────────────────────────
    # SAVE TO CSV
    # ──────────────────────────────────────────────────────────────

    def _save(self, df: pd.DataFrame, path: str, label: str):
        """Save DataFrame to CSV. Datetime formatted as YYYY-MM-DD."""
        CONFIG.paths.create_all_dirs()
        df_save = df.copy()
        df_save["Datetime"] = df_save["Datetime"].dt.strftime("%Y-%m-%d")
        df_save.to_csv(path, index=False)
        log.info(f"💾 Saved {label} → {path}  ({len(df_save):,} rows)")

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def fetch_nifty(self) -> pd.DataFrame:
        """
        Fetch NIFTY 50 daily OHLCV data.
        Symbol token and date range from CONFIG.

        Returns:
            DataFrame saved to CONFIG.paths.nifty_raw_path()
        """
        log.info("=" * 55)
        log.info("  Fetching NIFTY 50 Historical Data")
        log.info("=" * 55)

        # Cap end date to yesterday — Angel One rejects today/future dates (AB1012)
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        end = CONFIG.data.end_date or yesterday

        df = self._fetch_full(
            symbol_token = CONFIG.data.nifty_symbol,
            exchange     = CONFIG.data.exchange,
            interval     = CONFIG.data.timeframe,
            start_date   = CONFIG.data.start_date,
            end_date     = end,
            label        = "NIFTY50",
        )

        self._validate_and_preview(df, "NIFTY50")
        self._save(df, CONFIG.paths.nifty_raw_path(), "NIFTY50")
        return df

    def fetch_stock(self, symbol: str, token: str) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV data for a single stock.

        Args:
            symbol : Stock symbol string  (e.g. "RELIANCE")
            token  : Angel One symbol token (e.g. "2885")
                     Find tokens at:
                     https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json

        Returns:
            DataFrame saved to CONFIG.paths.stock_raw_path(symbol)
        """
        symbol = symbol.upper()
        log.info("=" * 55)
        log.info(f"  Fetching {symbol} Historical Data")
        log.info("=" * 55)

        # Cap end date to yesterday — Angel One rejects today/future dates (AB1012)
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        end = CONFIG.data.end_date or yesterday

        df = self._fetch_full(
            symbol_token = token,
            exchange     = CONFIG.data.exchange,
            interval     = CONFIG.data.timeframe,
            start_date   = CONFIG.data.start_date,
            end_date     = end,
            label        = symbol,
        )

        self._validate_and_preview(df, symbol)
        self._save(df, CONFIG.paths.stock_raw_path(symbol), symbol)
        return df

    def fetch_all(self, stocks: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch NIFTY + all stocks in one call.

        Args:
            stocks : dict of {symbol: token}
                     e.g. {"RELIANCE": "2885", "INFY": "1594"}

        Returns:
            dict of {symbol: DataFrame}
            Also includes "NIFTY50" key.

        Example:
            fetcher.fetch_all(stocks={"RELIANCE": "2885"})
        """
        results = {}

        # Always fetch NIFTY first (used as regime filter)
        results["NIFTY50"] = self.fetch_nifty()

        # Fetch each stock
        for symbol, token in stocks.items():
            try:
                results[symbol.upper()] = self.fetch_stock(symbol, token)
            except Exception as e:
                log.error(f"Failed to fetch {symbol}: {e}")

        log.info(f"\n✅ fetch_all complete | Fetched: {list(results.keys())}")
        return results


# ═══════════════════════════════════════════════════════════════════
# TOKEN LOOKUP HELPER
# ═══════════════════════════════════════════════════════════════════

def lookup_token(symbol: str, exchange: str = "NSE") -> Optional[str]:
    """
    Look up Angel One symbol token from the official scrip master JSON.
    Downloads fresh scrip master from Angel One CDN.

    Args:
        symbol   : e.g. "RELIANCE" (without -EQ suffix)
        exchange : "NSE" or "BSE"

    Returns:
        Token string if found, None otherwise.

    Usage:
        token = lookup_token("RELIANCE")
        print(token)  # "2885"
    """
    import requests

    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    log.info(f"Downloading Angel One scrip master...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        scrip_master = pd.DataFrame(response.json())

        # Filter: NSE equity segment, match symbol
        mask = (
            (scrip_master["exch_seg"] == exchange) &
            (scrip_master["symbol"]   == f"{symbol.upper()}-EQ")
        )
        match = scrip_master[mask]

        if match.empty:
            log.warning(f"Token not found for {symbol} on {exchange}.")
            return None

        token = str(match.iloc[0]["token"])
        log.info(f"✅ Token found: {symbol} → {token}")
        return token

    except Exception as e:
        log.error(f"Scrip master lookup failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT — python src/data/fetcher.py --symbol RELIANCE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Angel One Historical Data Fetcher")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Stock symbol to fetch (e.g. RELIANCE)")
    parser.add_argument("--token",  type=str, default=None,
                        help="Angel One token for stock (e.g. 2885). "
                             "If not provided, auto-lookup is attempted.")
    parser.add_argument("--nifty-only", action="store_true",
                        help="Fetch only NIFTY data")
    args = parser.parse_args()

    fetcher = AngelOneFetcher()

    if args.nifty_only:
        fetcher.fetch_nifty()

    elif args.symbol:
        token = args.token
        if not token:
            log.info(f"Token not provided — attempting auto-lookup for {args.symbol}...")
            token = lookup_token(args.symbol)
            if not token:
                raise ValueError(
                    f"Could not find token for {args.symbol}. "
                    f"Please provide it via --token."
                )
        fetcher.fetch_all(stocks={args.symbol: token})

    else:
        print("\nUsage examples:")
        print("  python src/data/fetcher.py --nifty-only")
        print("  python src/data/fetcher.py --symbol RELIANCE --token 2885")
        print("  python src/data/fetcher.py --symbol RELIANCE  # auto token lookup")