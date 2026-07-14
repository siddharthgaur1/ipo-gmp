"""
IPO GMP Seeder
Generates a realistic synthetic dataset of Indian IPO listings (2020–2025)
with GMP, subscription data, and actual listing returns.

This powers the ML model and backtester so the app works out-of-the-box
without needing live scraping. The patterns (GMP predicts listing ≈ 70%
directionally, subscription rate matters, SME IPOs are more volatile, etc.)
are modelled after real-world distributions from Chittorgarh data.

Run:
    python scripts/seed_data.py
"""

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # avoid UnicodeEncodeError on Windows' default cp1252 console

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ipo_gmp.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

CATEGORIES = ["Mainboard", "SME"]
SECTORS = [
    "Technology", "Financial Services", "Healthcare", "Consumer Goods",
    "Infrastructure", "Manufacturing", "Real Estate", "Chemicals",
    "Auto & Auto Components", "Retail", "Telecom", "Energy",
    "Agriculture", "Media & Entertainment", "Defence",
]
REGISTRARS = ["Link Intime", "KFin Technologies", "Bigshare Services", "Cameo Corporate"]

COMPANY_PREFIXES = [
    "Jyoti", "Bharat", "Sai", "Om", "Shri", "New Era", "Future",
    "Apex", "Pioneer", "Nova", "Crest", "Zenith", "Veda", "Param",
    "Stellar", "Horizon", "Anand", "Arjun", "Dev", "Primus",
]
COMPANY_SUFFIXES = [
    "Tech", "Infra", "Finance", "Industries", "Pharma", "Foods",
    "Logistics", "Energy", "Textiles", "Auto", "Healthcare", "Retail",
    "Solutions", "Systems", "Ventures", "Chemicals", "Projects",
]


def rand_company() -> str:
    return f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)} Ltd"


def rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def gen_ipo(idx: int) -> dict:
    cat = rng.choices(CATEGORIES, weights=[55, 45])[0]
    sector = rng.choice(SECTORS)

    issue_price = rng.randint(30, 1500) if cat == "Mainboard" else rng.randint(30, 300)
    lot_size = rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 200])
    issue_size_cr = rng.uniform(20, 5000) if cat == "Mainboard" else rng.uniform(5, 200)

    # Subscription rates (realistic: QIB usually highest)
    qib_sub = rng.uniform(0.1, 200)
    nii_sub = rng.uniform(0.1, 400)
    retail_sub = rng.uniform(0.2, 120)
    total_sub = (qib_sub + nii_sub * 0.35 + retail_sub * 0.35) / 1.7  # weighted approx

    # GMP as % of issue price — driven by fundamentals + noise
    # Positive bias when heavily subscribed, SME tends to have higher variance
    base_gmp_pct = (
        rng.gauss(0.20, 0.30) * min(total_sub / 50, 3)
        + rng.gauss(0, 0.05)  # sector noise
        + (0.10 if cat == "SME" else 0)  # SME premium
    )
    gmp_pct = max(-0.50, min(1.80, base_gmp_pct))
    gmp_rs = round(issue_price * gmp_pct, 1)
    gmp_expected_price = issue_price + gmp_rs

    # Listing return (correlated with GMP, but with noise)
    signal = gmp_pct * rng.uniform(0.5, 0.9)  # GMP captures 50–90% of move
    noise = rng.gauss(0, 0.12)  # ±12% random noise on listing
    listing_gain_pct = round((signal + noise) * 100, 2)

    # Clamp to realistic range
    listing_gain_pct = max(-60.0, min(200.0, listing_gain_pct))

    listing_price = round(issue_price * (1 + listing_gain_pct / 100), 2)

    # Dates
    open_d = rand_date(date(2020, 1, 1), date(2025, 10, 1))
    close_d = open_d + timedelta(days=rng.randint(2, 4))
    allot_d = close_d + timedelta(days=rng.randint(5, 8))
    listing_d = allot_d + timedelta(days=rng.randint(4, 7))

    # GMP at various stages (adds temporal realism)
    gmp_day1 = round(gmp_rs * rng.uniform(0.3, 0.7), 1)
    gmp_day2 = round(gmp_rs * rng.uniform(0.5, 0.9), 1)
    gmp_final = gmp_rs  # final GMP before listing

    return {
        "id": idx,
        "company": rand_company(),
        "category": cat,
        "sector": sector,
        "registrar": rng.choice(REGISTRARS),
        "issue_price": issue_price,
        "lot_size": lot_size,
        "issue_size_cr": round(issue_size_cr, 1),
        "open_date": open_d.isoformat(),
        "close_date": close_d.isoformat(),
        "allotment_date": allot_d.isoformat(),
        "listing_date": listing_d.isoformat(),
        "qib_subscription": round(qib_sub, 2),
        "nii_subscription": round(nii_sub, 2),
        "retail_subscription": round(retail_sub, 2),
        "total_subscription": round(total_sub, 2),
        "gmp_rs": gmp_rs,
        "gmp_pct": round(gmp_pct * 100, 2),
        "gmp_expected_price": round(gmp_expected_price, 2),
        "gmp_day1": gmp_day1,
        "gmp_day2": gmp_day2,
        "listing_price": listing_price,
        "listing_gain_pct": listing_gain_pct,
        "listing_day_high": round(listing_price * rng.uniform(1.0, 1.15), 2),
        "listing_day_low": round(listing_price * rng.uniform(0.88, 1.0), 2),
    }


def create_schema(conn: sqlite3.Connection):
    conn.executescript("""
    DROP TABLE IF EXISTS ipos;
    CREATE TABLE ipos (
        id                  INTEGER PRIMARY KEY,
        company             TEXT NOT NULL,
        category            TEXT NOT NULL,        -- Mainboard / SME
        sector              TEXT NOT NULL,
        registrar           TEXT,
        issue_price         REAL NOT NULL,
        lot_size            INTEGER NOT NULL,
        issue_size_cr       REAL NOT NULL,
        open_date           TEXT NOT NULL,
        close_date          TEXT NOT NULL,
        allotment_date      TEXT,
        listing_date        TEXT,
        qib_subscription    REAL,
        nii_subscription    REAL,
        retail_subscription REAL,
        total_subscription  REAL,
        gmp_rs              REAL,                 -- GMP in ₹ (pre-listing)
        gmp_pct             REAL,                 -- GMP as % of issue price
        gmp_expected_price  REAL,                 -- issue_price + gmp_rs
        gmp_day1            REAL,
        gmp_day2            REAL,
        listing_price       REAL,                 -- actual listing price
        listing_gain_pct    REAL,                 -- (listing - issue) / issue * 100
        listing_day_high    REAL,
        listing_day_low     REAL
    );
    CREATE INDEX idx_listing_date ON ipos(listing_date);
    CREATE INDEX idx_category     ON ipos(category);
    CREATE INDEX idx_sector       ON ipos(sector);
    """)


def seed(n: int = 800):
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    ipos = [gen_ipo(i + 1) for i in range(n)]
    conn.executemany("""
        INSERT INTO ipos VALUES (
            :id,:company,:category,:sector,:registrar,
            :issue_price,:lot_size,:issue_size_cr,
            :open_date,:close_date,:allotment_date,:listing_date,
            :qib_subscription,:nii_subscription,:retail_subscription,:total_subscription,
            :gmp_rs,:gmp_pct,:gmp_expected_price,:gmp_day1,:gmp_day2,
            :listing_price,:listing_gain_pct,:listing_day_high,:listing_day_low
        )
    """, ipos)
    conn.commit()
    n_total = conn.execute("SELECT COUNT(*) FROM ipos").fetchone()[0]
    conn.close()
    print(f"✅ Seeded {n_total} IPOs into {DB_PATH}")
    print(f"   Mainboard: {sum(1 for i in ipos if i['category']=='Mainboard')}")
    print(f"   SME:       {sum(1 for i in ipos if i['category']=='SME')}")
    avg_gain = sum(i['listing_gain_pct'] for i in ipos) / len(ipos)
    print(f"   Avg listing gain: {avg_gain:.1f}%")


if __name__ == "__main__":
    seed(800)
