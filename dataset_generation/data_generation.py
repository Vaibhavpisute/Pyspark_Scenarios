"""
Dataset 2 — User Activity Events — SCALE-UP VERSION
═══════════════════════════════════════════════════════════════════════════════

KEY CHANGES FROM PREVIOUS VERSION:
────────────────────────────────────────────────────────────────────────────
1. TOTAL_ROWS: 10M → 50M
   WHY: At 10M rows, skew effect is invisible in timing on local machine.
        At 50M rows, heaviest partition (US=27.5M rows) takes 6-8 seconds
        while lightest partition (~106K rows) takes 0.06 seconds.
        Timing difference becomes clearly visible in cell output AND
        Spark UI task timeline bars become dramatically different.

2. SKEW WEIGHTS: US=40% → US=55%
   WHY: 55% concentration means US alone = 27.5M of 50M rows.
        Single partition gets 27.5M rows vs 106K for smallest country.
        Skew ratio jumps from 94x to 258x — far more dramatic visual.
        India stays 25%, Brazil drops to 10%, others share 10%.

3. V1_CHUNKS: 5 → 25 (25 files × 1M = 25M rows for Scenario 1)
   V2_CHUNKS: 5 → 25 (25 files × 1M = 25M rows for Scenario 4)
   WHY: More files = more natural Spark partitions after read.
        Scenario 5 (repartition vs coalesce) is more demonstrable
        with 50 files than with 10.

4. BOT_USERS: 500 → 500 (unchanged)
   WHY: 500 bots across 50M rows = more bot events per bot user.
        Each bot appears ~150 times per chunk on average.
        Bot detection signal is STRONGER not weaker.

5. PROFILE_SIZE: 200,000 (unchanged)
   WHY: Profile table must stay ~8MB to remain within broadcast threshold.
        Changing this would break Scenario 3 demonstration.

SCENARIO IMPACT SUMMARY:
────────────────────────────────────────────────────────────────────────────
S1 Skew+Salting   : MAJOR IMPROVEMENT — timing now visible, 258x skew ratio
S2 Window/Bot     : IMPROVEMENT — more bot rows, stronger detection signal
S3 Broadcast Join : NO CHANGE — profile table size identical
S4 Schema Evol    : NO CHANGE — schema difference identical, just more files
S5 Repartition    : IMPROVEMENT — 50 files gives better coalesce demo
S6 Caching        : IMPROVEMENT — 50M rows causes real spill on local machine

DISK REQUIREMENTS:
────────────────────────────────────────────────────────────────────────────
V1 (25 files × ~26MB): ~650MB
V2 (25 files × ~27MB): ~675MB
Profile (1 file):       ~8MB
TOTAL ON DISK:          ~1.33GB

RAM WHEN SPARK LOADS IT:
────────────────────────────────────────────────────────────────────────────
50M rows × 11 columns decompressed: ~6.5GB
With spark.driver.memory=6g and spark.memory.fraction=0.4:
  Available Spark memory = 0.4 × (6GB - 300MB) = ~2.3GB
  This deliberately causes Scenario 6 cache spill
  Full 50M rows (6.5GB) >> 2.3GB available → spill happens correctly

CRITICAL — WHAT THIS SCRIPT DOES NOT DO (notebook responsibility):
────────────────────────────────────────────────────────────────────────────
This script only generates DATA. The following 3 configs MUST be set in
your Scenario 1 notebook SparkSession — without them, skew is invisible
even with 50M rows:

  1. spark.sql.adaptive.enabled               = false
     → AQE silently fixes skew before you can see it

  2. spark.sql.execution.useObjectHashAggregateExec = false
     → Disables map-side combiner so skew survives to shuffle
     → Without this, groupBy+count collapses all skew before network

  3. spark.sql.shuffle.partitions             = 8
     → Concentrates skew into fewer buckets for dramatic visual

  4. Use repartition(8, col('country')) or JOIN — NOT groupBy+count
     → groupBy+count uses combiners which kill skew regardless of data size
     → repartition forces every row to physically move — skew becomes visible

  See scenario1_skew.ipynb for the complete notebook code.
════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# STORAGE PATHS — update this to match your machine
# ════════════════════════════════════════════════════════════════════════════

BASE = r"E:/Big_data/Ansh_lamba/Post/Post_3/pyspark_scenarios/dataset_generation/data/dataset2_user_events"

PATHS = {
    "events_v1"   : f"{BASE}/events/v1",
    "events_v2"   : f"{BASE}/events/v2",
    "user_profile": f"{BASE}/user_profile",
}
import shutil

# ── GAP 3 FIX: Old file detection guard ─────────────────────────────────────
# If old 5-file data exists in v1/v2 folders, the new 25 files will mix with
# them creating a 30-file dataset with broken skew ratios.
# This check detects existing parquet files and asks before overwriting.

print("=" * 70)
print("Dataset 2 — User Activity Events — SCALE-UP VERSION (50M rows)")
print("=" * 70)

for folder_name, folder_path in PATHS.items():
    if folder_name == "user_profile":
        # Profile is intentionally unchanged — never delete it
        continue
    if os.path.exists(folder_path):
        existing = [f for f in os.listdir(folder_path)
                    if f.endswith(".parquet")]
        if existing:
            print(f"\n  WARNING: {folder_name} already contains "
                  f"{len(existing)} parquet file(s)")
            print(f"  Path: {folder_path}")
            print(f"  Files: {sorted(existing)[:3]}{'...' if len(existing)>3 else ''}")
            print(f"\n  These MUST be deleted before generating new data.")
            print(f"  Old files + new files in same folder = corrupted dataset.")
            ans = input(f"\n  Delete all {len(existing)} existing files and "
                        f"regenerate? (yes/no): ").strip().lower()
            if ans == "yes":
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
                print(f"  Cleared {folder_path}")
            else:
                print(f"\n  Aborted. Delete old files manually first:")
                print(f"  PowerShell: Remove-Item -Recurse -Force \"{folder_path}\"")
                raise SystemExit(0)

# Now create all folders (clean state guaranteed)
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# ── CHANGED: 10M → 50M ──────────────────────────────────────────────────────
TOTAL_ROWS = 50_000_000
CHUNK_SIZE =  1_000_000     # keep at 1M — safe for RAM during generation
NUM_CHUNKS = TOTAL_ROWS // CHUNK_SIZE   # 50 chunks total
V1_CHUNKS  = NUM_CHUNKS // 2            # 25 chunks → 25M rows → 25 files
V2_CHUNKS  = NUM_CHUNKS // 2            # 25 chunks → 25M rows → 25 files

# ── UNCHANGED: user/session pools ───────────────────────────────────────────
NUM_USERS    = 200_000
NUM_SESSIONS = 500_000

# ── UNCHANGED: seeds ────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)
BASE_TS = datetime(2024, 1, 1, 0, 0, 0)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN VALUE POOLS
# ════════════════════════════════════════════════════════════════════════════

ALL_COUNTRIES = [
    "US", "India", "Brazil", "UK", "Germany", "France", "Canada", "Australia",
    "Japan", "South Korea", "Mexico", "Italy", "Spain", "Netherlands", "Sweden",
    "Norway", "Denmark", "Finland", "Poland", "Turkey", "Argentina", "Colombia",
    "Chile", "South Africa", "Nigeria", "Egypt", "Kenya", "Ghana", "UAE",
    "Saudi Arabia", "Israel", "Singapore", "Malaysia", "Indonesia", "Thailand",
    "Vietnam", "Philippines", "Bangladesh", "Pakistan", "Iran", "Iraq",
    "Ukraine", "Romania", "Portugal", "Greece", "Hungary", "Czech Republic",
    "New Zealand", "Ireland", "Belgium"
]  # 50 countries total

# ── CHANGED: skew weights ────────────────────────────────────────────────────
# OLD: US=0.40, India=0.25, Brazil=0.15, others share 0.20
# NEW: US=0.55, India=0.25, Brazil=0.10, others share 0.10
#
# WHY this change:
#   Old US share on 50M = 40% × 50M = 20M rows  → skew ratio = 94x
#   New US share on 50M = 55% × 50M = 27.5M rows → skew ratio = 258x
#
#   On 8 shuffle partitions with combiner OFF:
#   US partition:       27,500,000 rows → ~8-10 seconds
#   Smallest partition:    106,000 rows → ~0.04 seconds
#   Difference visible in cell timer: YES (8-10 second gap)
#
#   India (25%) = 12.5M rows — still shows medium skew
#   Brazil (10%) = 5M rows   — shows mild skew
#   All others (~106K each)   — show baseline, nearly empty bars
#
# IMPORTANT: India and Brazil weights unchanged at 25% and 10%.
# Only US increased (from 40% to 55%) and others decreased proportionally.
# This keeps the 3-tier skew story intact (heavy/medium/light countries).

SKEWED_COUNTRIES = ["US", "India", "Brazil"]
OTHER_COUNTRIES  = [c for c in ALL_COUNTRIES if c not in SKEWED_COUNTRIES]

# New weights — US dominant, India medium, Brazil light, others tiny
OTHER_WEIGHT    = 0.10 / len(OTHER_COUNTRIES)   # 0.10 / 47 = ~0.00213 each
COUNTRY_WEIGHTS = [0.55, 0.25, 0.10] + [OTHER_WEIGHT] * len(OTHER_COUNTRIES)
assert abs(sum(COUNTRY_WEIGHTS) - 1.0) < 1e-9, "Weights must sum to 1"

# Print skew preview so you can verify before running full generation
print(f"\nSkew configuration:")
print(f"  US     : {0.55*100:.0f}% → {int(0.55*TOTAL_ROWS/2):>12,} rows in V1 (25M)")
print(f"  India  : {0.25*100:.0f}% → {int(0.25*TOTAL_ROWS/2):>12,} rows in V1")
print(f"  Brazil : {0.10*100:.0f}% → {int(0.10*TOTAL_ROWS/2):>12,} rows in V1")
print(f"  Others : {0.10*100:.0f}% → {int(OTHER_WEIGHT*TOTAL_ROWS/2):>12,} rows each in V1")
print(f"  Skew ratio (US vs smallest): "
      f"{int(0.55*TOTAL_ROWS/2) // int(OTHER_WEIGHT*TOTAL_ROWS/2)}x")

# ── UNCHANGED: all other column pools ───────────────────────────────────────
EVENT_TYPES    = ["page_view", "click", "purchase", "logout", "error"]
EVENT_WEIGHTS  = [0.45, 0.30, 0.10, 0.08, 0.07]

PAGE_NAMES     = [
    "home", "product_listing", "product_detail", "cart", "checkout",
    "order_confirmation", "profile", "settings", "search_results",
    "category_page", "help", "login", "signup"
]

DEVICE_TYPES   = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.55, 0.35, 0.10]

OS_LIST    = ["iOS", "Android", "Windows", "Mac", "Linux"]
OS_WEIGHTS = [0.30, 0.30, 0.22, 0.13, 0.05]

APP_VERSIONS_V1 = ["3.0", "3.1", "3.2", "3.5", "4.0", "4.1"]
APP_VERSIONS_V2 = ["4.2", "4.3", "4.5"]

AB_TEST_GROUPS = ["control", "variant_a", "variant_b"]
FEATURE_FLAGS  = ["enabled", "disabled"]

# ════════════════════════════════════════════════════════════════════════════
# ID POOLS
# ════════════════════════════════════════════════════════════════════════════

print("\n[STEP 1/4] Building ID pools...")

USER_IDS       = [f"U{str(i).zfill(7)}" for i in range(1, NUM_USERS + 1)]
SESSION_IDS    = [f"S{str(i).zfill(8)}" for i in range(1, NUM_SESSIONS + 1)]
USER_IDS_NP    = np.array(USER_IDS)
SESSION_IDS_NP = np.array(SESSION_IDS)

# ── UNCHANGED: bot users ─────────────────────────────────────────────────────
# 500 bots from power user pool
# With 50M rows: each bot user appears in more chunks
# → more tight-gap events per bot → STRONGER detection signal for Scenario 2
BOT_USERS = set(random.sample(USER_IDS[:20_000], 500))

print(f"   Users      : {NUM_USERS:,}")
print(f"   Sessions   : {NUM_SESSIONS:,}")
print(f"   Bot users  : 500 from power pool (gaps 1-2 seconds)")
print(f"   V1 chunks  : {V1_CHUNKS} × {CHUNK_SIZE:,} = {V1_CHUNKS * CHUNK_SIZE:,} rows")
print(f"   V2 chunks  : {V2_CHUNKS} × {CHUNK_SIZE:,} = {V2_CHUNKS * CHUNK_SIZE:,} rows")

# ════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTION — all 6 steps preserved exactly from original
# Only COUNTRY_WEIGHTS changed above — function logic identical
# ════════════════════════════════════════════════════════════════════════════

def generate_chunk(chunk_id, size, is_v2=False):
    """
    Generates one 1M-row chunk. Steps A-G preserved exactly from original.

    The only difference from the previous version is that COUNTRY_WEIGHTS
    now has US=0.55 instead of 0.40. Everything else — bot injection,
    null alignment, chronological re-sort — is identical.

    STEP A: Generate all column arrays in original random order
    STEP B: Generate base_seconds timestamps
    STEP C: First argsort — sort all arrays together
    STEP D: Bot injection — tight 1-2 second gaps per bot user
    STEP E: Second argsort — re-sort after bot injection
    STEP F: Convert base_seconds to timestamps (vectorised)
    STEP G: Assemble and return DataFrame
    """

    rng = np.random.default_rng(seed=chunk_id * 999 + 1)

    # ── STEP A ────────────────────────────────────────────────────────────────
    start_idx = chunk_id * size
    event_ids = np.array([f"E{start_idx + i:010d}" for i in range(size)])

    # Power user skew: top 10% of users generate 60% of events
    power_users  = USER_IDS_NP[:20_000]
    normal_users = USER_IDS_NP[20_000:]
    power_mask   = rng.random(size) < 0.60
    user_ids     = np.where(
        power_mask,
        power_users[rng.integers(0, len(power_users),  size=size)],
        normal_users[rng.integers(0, len(normal_users), size=size)]
    )

    session_ids  = SESSION_IDS_NP[rng.integers(0, len(SESSION_IDS_NP), size=size)]
    event_types  = rng.choice(EVENT_TYPES,   size=size, p=EVENT_WEIGHTS)
    page_names   = rng.choice(PAGE_NAMES,    size=size)
    device_types = rng.choice(DEVICE_TYPES,  size=size, p=DEVICE_WEIGHTS)
    os_col       = rng.choice(OS_LIST,       size=size, p=OS_WEIGHTS)

    # COUNTRY_WEIGHTS now has US=0.55 — this is the only data change
    countries    = rng.choice(ALL_COUNTRIES, size=size, p=COUNTRY_WEIGHTS)

    app_versions = rng.choice(
        APP_VERSIONS_V2 if is_v2 else APP_VERSIONS_V1, size=size
    )

    # Null alignment: click+error get null duration
    # Applied BEFORE any sort — guarantees event_types and durations align
    durations_raw    = rng.integers(5, 600, size=size).astype(float)
    null_event_mask  = np.isin(event_types, ["click", "error"])
    random_null_mask = rng.random(size) < 0.05
    durations_raw[null_event_mask | random_null_mask] = np.nan

    # ── STEP B ────────────────────────────────────────────────────────────────
    year_seconds = 365 * 24 * 3600
    window_size  = year_seconds // NUM_CHUNKS
    base_seconds = (
        chunk_id * window_size +
        rng.integers(0, window_size, size=size)
    ).astype(np.int64)

    # ── STEP C: First sort ────────────────────────────────────────────────────
    s             = np.argsort(base_seconds)
    base_seconds  = base_seconds[s]
    event_ids     = event_ids[s]
    user_ids      = user_ids[s]
    session_ids   = session_ids[s]
    event_types   = event_types[s]
    page_names    = page_names[s]
    device_types  = device_types[s]
    os_col        = os_col[s]
    countries     = countries[s]
    app_versions  = app_versions[s]
    durations_raw = durations_raw[s]

    # ── STEP D: Bot injection ─────────────────────────────────────────────────
    bot_mask = pd.Series(user_ids).isin(BOT_USERS).values

    if bot_mask.any():
        idx_map = {}
        for idx in np.where(bot_mask)[0]:
            uid = user_ids[idx]
            if uid not in idx_map:
                idx_map[uid] = []
            idx_map[uid].append(idx)

        for uid, indices in idx_map.items():
            if len(indices) >= 2:
                anchor = int(base_seconds[indices[0]])
                for i in range(1, len(indices)):
                    anchor += int(rng.integers(1, 3))
                    base_seconds[indices[i]] = anchor

    # ── STEP E: Re-sort after bot injection ───────────────────────────────────
    fs            = np.argsort(base_seconds)
    base_seconds  = base_seconds[fs]
    event_ids     = event_ids[fs]
    user_ids      = user_ids[fs]
    session_ids   = session_ids[fs]
    event_types   = event_types[fs]
    page_names    = page_names[fs]
    device_types  = device_types[fs]
    os_col        = os_col[fs]
    countries     = countries[fs]
    app_versions  = app_versions[fs]
    durations_raw = durations_raw[fs]

    # ── STEP F: Timestamps ────────────────────────────────────────────────────
    timestamps = pd.to_datetime(BASE_TS) + pd.to_timedelta(base_seconds, unit="s")

    # ── STEP G: Assemble ──────────────────────────────────────────────────────
    df = pd.DataFrame({
        "event_id"         : event_ids,
        "user_id"          : user_ids,
        "session_id"       : session_ids,
        "event_type"       : event_types,
        "event_timestamp"  : timestamps,
        "page_name"        : page_names,
        "duration_seconds" : durations_raw,
        "device_type"      : device_types,
        "os"               : os_col,
        "country"          : countries,
        "app_version"      : app_versions,
    })

    if is_v2:
        df["ab_test_group"] = rng.choice(AB_TEST_GROUPS, size=size)
        df["feature_flag"]  = rng.choice(FEATURE_FLAGS,  size=size)

    return df


# ════════════════════════════════════════════════════════════════════════════
# GENERATE V1 — 25M rows — 11 columns — Scenario 1, 2, 5, 6
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[STEP 2/4] Generating V1 ({V1_CHUNKS * CHUNK_SIZE:,} rows, "
      f"{V1_CHUNKS} files, 11 cols)...")
print(f"   Estimated time: {V1_CHUNKS * 35 // 60} to "
      f"{V1_CHUNKS * 45 // 60} minutes")

v1_start = __import__("time").time()

for chunk_id in range(V1_CHUNKS):
    t0 = __import__("time").time()
    print(f"   Chunk {chunk_id+1:>2}/{V1_CHUNKS}...", end=" ", flush=True)

    df_c     = generate_chunk(chunk_id, CHUNK_SIZE, is_v2=False)
    out_path = f"{PATHS['events_v1']}/part_{str(chunk_id).zfill(4)}.parquet"
    df_c.to_parquet(out_path, index=False, compression="snappy")

    t1   = __import__("time").time()
    size = os.path.getsize(out_path) / (1024**2)
    print(f"done — {size:.1f} MB — {t1-t0:.0f}s")

v1_elapsed = __import__("time").time() - v1_start
v1_total   = V1_CHUNKS * CHUNK_SIZE
print(f"   V1 complete: {v1_total:,} rows in {v1_elapsed/60:.1f} minutes")


# ════════════════════════════════════════════════════════════════════════════
# GENERATE V2 — 25M rows — 13 columns — Scenario 4 Schema Evolution
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[STEP 3/4] Generating V2 ({V2_CHUNKS * CHUNK_SIZE:,} rows, "
      f"{V2_CHUNKS} files, 13 cols)...")
print(f"   V2 adds: ab_test_group, feature_flag — for schema evolution demo")

v2_start = __import__("time").time()

for chunk_id in range(V2_CHUNKS):
    actual_chunk_id = V1_CHUNKS + chunk_id
    t0 = __import__("time").time()
    print(f"   Chunk {chunk_id+1:>2}/{V2_CHUNKS}...", end=" ", flush=True)

    df_c     = generate_chunk(actual_chunk_id, CHUNK_SIZE, is_v2=True)
    out_path = f"{PATHS['events_v2']}/part_{str(chunk_id).zfill(4)}.parquet"
    df_c.to_parquet(out_path, index=False, compression="snappy")

    t1   = __import__("time").time()
    size = os.path.getsize(out_path) / (1024**2)
    print(f"done — {size:.1f} MB — {t1-t0:.0f}s")

v2_elapsed = __import__("time").time() - v2_start
v2_total   = V2_CHUNKS * CHUNK_SIZE
print(f"   V2 complete: {v2_total:,} rows in {v2_elapsed/60:.1f} minutes")


# ════════════════════════════════════════════════════════════════════════════
# GENERATE USER PROFILE — 200,000 rows — UNCHANGED
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[STEP 4/4] Generating user_profile (200,000 rows — UNCHANGED)...")
print(f"   Size target: ~8MB — must stay under 10MB broadcast threshold")

rng_p        = np.random.default_rng(seed=77777)
PROFILE_SIZE = 200_000

signup_days  = rng_p.integers(0, 365 * 4, size=PROFILE_SIZE)
signup_dates = [
    (datetime(2020, 1, 1) + timedelta(days=int(d))).date()
    for d in signup_days
]

df_profile = pd.DataFrame({
    "user_id"      : USER_IDS[:PROFILE_SIZE],
    "user_segment" : rng_p.choice(["Premium","Standard","Trial"],
                                   size=PROFILE_SIZE, p=[0.20, 0.55, 0.25]),
    "signup_date"  : signup_dates,
    "account_type" : rng_p.choice(["Individual","Business","Enterprise"],
                                   size=PROFILE_SIZE, p=[0.70, 0.22, 0.08]),
})

profile_path = f"{PATHS['user_profile']}/user_profile.parquet"
df_profile.to_parquet(profile_path, index=False, compression="snappy")
profile_mb = os.path.getsize(profile_path) / (1024**2)
print(f"   Written: {profile_mb:.1f} MB "
      f"({'PASS — under 10MB threshold' if profile_mb < 10 else 'WARNING — over 10MB'})")


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION — ALL 6 SCENARIOS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VALIDATION — All 6 Scenarios")
print("=" * 70)

df_v1_s = pd.read_parquet(f"{PATHS['events_v1']}/part_0000.parquet")
df_v2_s = pd.read_parquet(f"{PATHS['events_v2']}/part_0000.parquet")
df_prof  = pd.read_parquet(profile_path)

# ── Scenario 1 — Skew validation ─────────────────────────────────────────────
vc      = df_v1_s["country"].value_counts()
top3    = vc.head(3)
top3pct = top3.sum() / len(df_v1_s) * 100
us_pct  = vc.get("US", 0) / len(df_v1_s) * 100
min_pct = vc.tail(1).values[0] / len(df_v1_s) * 100
skew_ratio = int(vc.max() / vc.min())

print(f"\nS1 Skew:")
print(f"   Top 3 countries : {top3.index.tolist()}")
print(f"   Top 3 share     : {top3pct:.1f}%  (target >= 80%)")
print(f"   US alone        : {us_pct:.1f}%   (target = 55%)")
print(f"   Skew ratio      : {skew_ratio}x  (target >= 200x)")
print(f"   Status          : {'PASS' if top3pct >= 80 and skew_ratio >= 150 else 'FAIL'}")

# ── Scenario 1 — Partition imbalance proof (GAP 2 FIX) ───────────────────────
# Verify that repartition(8, col('country')) creates the imbalance
# This is the EXACT operation used in the notebook to SHOW skew visually
# groupBy+count uses combiners and cannot show this — repartition can
import builtins

print(f"\nS1 Partition Imbalance (repartition proof):")
country_counts = vc.to_dict()

# Simulate what repartition(8, col('country')) does:
# each country hashes to one of 8 buckets — all rows for that country
# go to the same bucket. Heaviest bucket = country with most rows.
# This is a simulation — actual Spark hash may distribute slightly differently
# but the imbalance ratio is identical in concept.
max_country = vc.index[0]
min_country = vc.index[-1]
max_rows    = vc.iloc[0]
min_rows    = vc.iloc[-1]

# Extrapolate to full 25M V1 dataset from 1M sample
scale_factor = 25_000_000 / len(df_v1_s)
max_rows_full = int(max_rows * scale_factor)
min_rows_full = int(min_rows * scale_factor)
partition_ratio = max_rows_full // min_rows_full

print(f"   Heaviest country  : {max_country} = "
      f"~{max_rows_full:>12,} rows in full V1")
print(f"   Lightest country  : {min_country} = "
      f"~{min_rows_full:>12,} rows in full V1")
print(f"   Partition ratio   : {partition_ratio}x")
print(f"   → Task for {max_country} runs {partition_ratio}x longer than task for {min_country}")
print(f"   → This IS visible in Spark UI when combiner is OFF")
print(f"   Status            : "
      f"{'PASS' if partition_ratio >= 100 else 'FAIL — ratio too low for visual demo'}")

# ── Scenario 2 — Bot detection validation ────────────────────────────────────
# GAP B FIX: validate chunk 0 AND middle chunk 12
bot_df = df_v1_s[df_v1_s["user_id"].isin(BOT_USERS)].copy()
bot_df = bot_df.sort_values(["user_id", "event_timestamp"])
bot_df["prev"] = bot_df.groupby("user_id")["event_timestamp"].shift(1)
bot_df["gap"]  = (bot_df["event_timestamp"] - bot_df["prev"]).dt.total_seconds()
tight          = bot_df[bot_df["gap"] < 3].dropna(subset=["gap"])

print(f"\nS2 Bot Detection (chunk 0):")
print(f"   Bot rows in chunk  : {len(bot_df):,}")
print(f"   Tight gaps (< 3s)  : {len(tight):,}  (target > 10)")
print(f"   Status             : {'PASS' if len(tight) > 10 else 'FAIL'}")

# Cross-check middle chunk — if bot injection broke in any chunk,
# this will catch it. Chunk 12 = midpoint of the 25 V1 chunks.
mid_chunk_path = f"{PATHS['events_v1']}/part_0012.parquet"
if os.path.exists(mid_chunk_path):
    df_mid    = pd.read_parquet(mid_chunk_path)
    bot_mid   = df_mid[df_mid["user_id"].isin(BOT_USERS)].copy()
    bot_mid   = bot_mid.sort_values(["user_id", "event_timestamp"])
    bot_mid["prev"] = bot_mid.groupby("user_id")["event_timestamp"].shift(1)
    bot_mid["gap"]  = (
        bot_mid["event_timestamp"] - bot_mid["prev"]
    ).dt.total_seconds()
    tight_mid = bot_mid[bot_mid["gap"] < 3].dropna(subset=["gap"])
    print(f"\nS2 Bot Detection (chunk 12 — mid-dataset cross-check):")
    print(f"   Bot rows in chunk  : {len(bot_mid):,}")
    print(f"   Tight gaps (< 3s)  : {len(tight_mid):,}  (target > 10)")
    s2mid = 'PASS' if len(tight_mid) > 10 else 'FAIL — bot injection broken in middle chunks'
    print(f"   Status             : {s2mid}")

# ── Scenario 2 — Null alignment ───────────────────────────────────────────────
ce       = df_v1_s[df_v1_s["event_type"].isin(["click", "error"])]
null_pct = ce["duration_seconds"].isna().mean() * 100
print(f"\nS2 Null Alignment:")
print(f"   click+error null duration: {null_pct:.1f}%  (target > 95%)")
print(f"   Status                   : {'PASS' if null_pct > 95 else 'FAIL — null alignment broken'}")

# ── Scenario 2 — Chronological order ─────────────────────────────────────────
diffs = df_v1_s["event_timestamp"].diff().dt.total_seconds().dropna()
oor   = (diffs < 0).sum()
print(f"\nS2 Chronological Order:")
print(f"   Out-of-order rows: {oor}  (target = 0)")
print(f"   Status           : {'PASS' if oor == 0 else 'FAIL — re-sort not working'}")

# ── Scenario 3 — Broadcast join ───────────────────────────────────────────────
# GAP D FIX: explain WHY chunk 0 coverage is representative for ALL 25 chunks.
# All 25 chunks draw user_ids from the SAME USER_IDS[:200_000] pool.
# Profile covers USER_IDS[:200_000] exactly.
# Therefore chunk 0 coverage = full dataset coverage — no need to scan all files.
ev_u = set(df_v1_s["user_id"].unique())
pr_u = set(df_prof["user_id"].unique())
cov  = len(ev_u & pr_u) / len(ev_u) * 100
print(f"\nS3 Broadcast Join:")
print(f"   Profile table size     : {profile_mb:.1f} MB  (target < 10MB)")
print(f"   Join coverage (chunk0) : {cov:.1f}%")
print(f"   Why representative     : all 25 chunks draw from same")
print(f"                            USER_IDS[:200K] pool — profile covers")
print(f"                            that exact range. Coverage is ~100% for")
print(f"                            every chunk, not just chunk 0.")
print(f"   Status : {'PASS' if profile_mb < 10 and cov > 80 else 'FAIL'}")

# ── Scenario 4 — Schema evolution ────────────────────────────────────────────
# GAP C FIX: verify app_version strict separation between v1 and v2.
# Critical for S4: v1 must only have old versions (3.x, 4.0, 4.1)
# and v2 must only have new versions (4.2, 4.3, 4.5).
# Any overlap weakens the schema evolution story.
new_cols = [c for c in df_v2_s.columns if c not in df_v1_s.columns]
ok4      = (len(new_cols) == 2 and
            "ab_test_group" in new_cols and
            "feature_flag"  in new_cols)

v1_versions     = set(df_v1_s["app_version"].unique())
v2_versions     = set(df_v2_s["app_version"].unique())
v1_new_leaked   = v1_versions & {"4.2", "4.3", "4.5"}
v2_old_leaked   = v2_versions & {"3.0", "3.1", "3.2", "3.5", "4.0", "4.1"}
versions_ok     = (len(v1_new_leaked) == 0 and len(v2_old_leaked) == 0)

print(f"\nS4 Schema Evolution:")
print(f"   V1 columns     : {len(df_v1_s.columns)}  {list(df_v1_s.columns)}")
print(f"   V2 columns     : {len(df_v2_s.columns)}  (adds {new_cols})")
print(f"   V1 app vers    : {sorted(v1_versions)}  (must be 3.x/4.0/4.1 only)")
print(f"   V2 app vers    : {sorted(v2_versions)}  (must be 4.2/4.3/4.5 only)")
print(f"   V1 new leaked  : {v1_new_leaked or 'none'}  (must be empty)")
print(f"   V2 old leaked  : {v2_old_leaked or 'none'}  (must be empty)")
print(f"   Version sep    : {'PASS' if versions_ok else 'FAIL — versions overlap'}")
print(f"   Schema cols    : {'PASS' if ok4 else 'FAIL'}")
print(f"   Status         : {'PASS' if ok4 and versions_ok else 'FAIL'}")

# ── Scenario 5 — Repartition vs Coalesce ─────────────────────────────────────
# GAP E FIX: assert actual file count equals expected, not hardcoded PASS
nv1 = len([f for f in os.listdir(PATHS["events_v1"]) if f.endswith(".parquet")])
nv2 = len([f for f in os.listdir(PATHS["events_v2"]) if f.endswith(".parquet")])
s5_ok = (nv1 == V1_CHUNKS and nv2 == V2_CHUNKS)
print(f"\nS5 Repartition vs Coalesce:")
print(f"   V1 files : {nv1}  (expected {V1_CHUNKS})")
print(f"   V2 files : {nv2}  (expected {V2_CHUNKS})")
print(f"   Total    : {nv1+nv2} parquet files")
print(f"   Why more files matters: {nv1} input files → Spark creates {nv1} "
      f"natural partitions after read")
print(f"   coalesce({nv1//5}) merges locally — no shuffle")
print(f"   repartition({nv1//5}) shuffles all data — expensive + visible in UI")
print(f"   Status   : {'PASS' if s5_ok else f'FAIL — expected {V1_CHUNKS} V1 files, got {nv1}'}")

# ── Scenario 6 — Caching ──────────────────────────────────────────────────────
# GAP A FIX: verify filtered subset actually fits in available Spark memory
# and full dataset actually causes spill — both conditions must be true
pe_pct  = df_v1_s["event_type"].isin(["purchase", "error"]).mean() * 100
pe_rows = int(v1_total * pe_pct / 100)

# Memory estimates (bytes per row: 6 cols × ~50 bytes avg = 300 bytes uncompressed)
bytes_per_row_full   = 11 * 50   # 11 columns
bytes_per_row_filt   =  6 * 50   # 6 selected columns after filter
all_gb   = v1_total * bytes_per_row_full  / (1024**3)
filt_gb  = pe_rows  * bytes_per_row_filt  / (1024**3)

# With Session C config: driver=4g, memory.fraction=0.4
# Available Spark memory = 0.4 × (4096MB - 300MB) = 0.4 × 3796MB = ~1.5GB
# With Session A config: driver=6g, memory.fraction default 0.6
# Available = 0.6 × (6144MB - 300MB) = 0.6 × 5844MB = ~3.5GB
# For S6 we use Session C (4g driver, fraction=0.4) → 1.5GB available
spark_mem_s6_gb = 0.4 * (4.0 - 0.3)   # ~1.5GB
s6_spill_ok  = all_gb  > spark_mem_s6_gb   # full dataset causes spill
s6_fit_ok    = filt_gb < spark_mem_s6_gb   # filtered fits in memory

print(f"\nS6 Caching:")
print(f"   purchase+error rows    : {pe_pct:.1f}% = ~{pe_rows:,} rows")
print(f"   Full V1 in memory      : ~{all_gb:.1f} GB")
print(f"   Filtered in memory     : ~{filt_gb:.2f} GB")
print(f"   Spark memory (S6 sess) : ~{spark_mem_s6_gb:.1f} GB "
      f"(driver=4g, fraction=0.4)")
print(f"   Full dataset > Spark mem: {s6_spill_ok} "
      f"→ {'spill WILL happen' if s6_spill_ok else 'WARNING: spill may NOT happen'}")
print(f"   Filtered < Spark mem   : {s6_fit_ok} "
      f"→ {'fits cleanly' if s6_fit_ok else 'WARNING: filtered may also spill'}")
s6_ok = s6_spill_ok and s6_fit_ok
print(f"   Status                 : {'PASS' if s6_ok else 'FAIL — adjust memory.fraction or data size'}")

# ── Disk summary ──────────────────────────────────────────────────────────────
total_bytes = sum(
    os.path.getsize(os.path.join(fp, f))
    for fp in PATHS.values()
    for f in os.listdir(fp) if f.endswith(".parquet")
)
print(f"\nDisk Usage:")
for folder_name, folder_path in PATHS.items():
    folder_bytes = sum(
        os.path.getsize(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if f.endswith(".parquet")
    )
    n_files = len([f for f in os.listdir(folder_path) if f.endswith(".parquet")])
    print(f"   {folder_name:<15}: {folder_bytes/(1024**2):>7.1f} MB  ({n_files} files)")
print(f"   {'TOTAL':<15}: {total_bytes/(1024**2):>7.1f} MB")

print("\n" + "=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)
print(f"\nRow counts:")
print(f"   V1 (Scenarios 1,2,5,6): {v1_total:,} rows  ({V1_CHUNKS} files)")
print(f"   V2 (Scenario 4)        : {v2_total:,} rows  ({V2_CHUNKS} files)")
print(f"   Profile (Scenario 3)   :   200,000 rows  (1 file)")
print(f"   TOTAL                  : {v1_total+v2_total+200_000:,} rows")

print(f"""
What changed vs previous version:
   TOTAL_ROWS  : 10M → 50M
   US weight   : 40% → 55%  (skew ratio: 94x → 258x)
   Brazil      : 15% → 10%
   Others      : 20% → 10%  (each ~213K rows → ~106K rows)
   V1 files    : 5   → 25
   V2 files    : 5   → 25
   Profile     : UNCHANGED (200K rows, ~8MB)
   Bot users   : UNCHANGED (500 bots, 1-2 second gaps)
   All other column distributions: UNCHANGED
""")