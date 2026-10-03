"""
scripts/migrate_sqlite_to_supabase.py
One-time copy of the local SQLite database (cedula.db) into Supabase Postgres.

  1. Backs up the SQLite file to backups/cedula_<timestamp>.db
  2. Copies every row, keeping the original ids, in foreign-key order
  3. Uploads profile pictures and referenced CTC PDFs to Supabase Storage
     (only when SUPABASE_URL / SUPABASE_SECRET_KEY are set)
  4. Resets the identity sequences so new rows continue after the old ids
  5. Prints a per-table row-count comparison

The schema must already exist in Supabase (it is created by the Supabase
migrations, not by this script). The target tables must be empty; the whole
copy runs in one transaction, so a failure leaves Supabase unchanged.

Usage (from capstone_backend/, with DATABASE_URL in .env pointing at Supabase):
    venv\\Scripts\\python -m scripts.migrate_sqlite_to_supabase
    venv\\Scripts\\python -m scripts.migrate_sqlite_to_supabase --pdfs-only
        (re-upload PDFs that were saved locally because an upload failed)
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import date, datetime, timezone

from sqlalchemy import create_engine, text

from app.core.config import settings
from app.services import storage_service

# Parents before children so foreign keys are satisfied.
TABLES = [
    "staff_users",
    "citizen_users",
    "applicant_records",
    "ctc_issuance",
    "mobile_submissions",
]


def backup_sqlite(source: str) -> str:
    os.makedirs("backups", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join("backups", f"cedula_{stamp}.db")
    # sqlite3's backup API gives a consistent copy even if the server is running.
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    return target


def target_column_types(conn, table: str) -> dict:
    rows = conn.execute(
        text(
            "select column_name, data_type from information_schema.columns "
            "where table_schema = 'public' and table_name = :t"
        ),
        {"t": table},
    )
    return {name: data_type for name, data_type in rows}


def parse_timestamp(value):
    if value is None or isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(str(value))
    # Every timestamp in cedula.db was written in UTC (CURRENT_TIMESTAMP,
    # utcnow(), now(timezone.utc)) but stored without an offset.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def convert(value, data_type: str):
    if value is None:
        return None
    if data_type == "timestamp with time zone":
        return parse_timestamp(value)
    if data_type == "date":
        return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
    if data_type == "boolean":
        return bool(value)
    if data_type == "jsonb":
        return value if isinstance(value, str) else json.dumps(value)
    if data_type == "double precision":
        return float(value)
    return value


def upload_profile_picture(row: dict) -> None:
    value = row.get("profile_picture")
    if not storage_service.is_enabled() or not storage_service.decode_image_data_url(value):
        return
    # Fixed object name so re-running the script overwrites instead of duplicating.
    row["profile_picture"] = storage_service.store_profile_picture(row["id"], value, object_name="migrated")
    print(f"    citizen {row['id']}: profile picture -> {row['profile_picture']}")


def upload_pdf(row: dict) -> None:
    path = row.get("pdf_path")
    if not storage_service.is_enabled() or not path or storage_service.is_storage_ref(path):
        return
    local = path.replace("\\", os.sep).replace("/", os.sep)
    if not os.path.isfile(local):
        print(f"    WARNING ctc {row.get('ctc_number')}: PDF not found at {path}, keeping path")
        return
    row["pdf_path"] = storage_service.store_pdf(local)
    print(f"    ctc {row.get('ctc_number')}: PDF -> {row['pdf_path']}")


def migrate(source: str, engine) -> None:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row

    with engine.begin() as conn:
        non_empty = [
            t for t in TABLES
            if conn.execute(text(f'select count(*) from public."{t}"')).scalar()
        ]
        if non_empty:
            sys.exit(f"Target tables are not empty: {', '.join(non_empty)}. Aborting.")

        for table in TABLES:
            types = target_column_types(conn, table)
            rows = [dict(r) for r in src.execute(f'select * from "{table}" order by id')]
            print(f"  {table}: {len(rows)} rows")
            if not rows:
                continue
            columns = [c for c in rows[0].keys() if c in types]
            skipped = set(rows[0].keys()) - set(columns)
            if skipped:
                sys.exit(f"{table}: columns missing in Supabase: {sorted(skipped)}")

            placeholders = ", ".join(
                f"cast(:{c} as jsonb)" if types[c] == "jsonb" else f":{c}" for c in columns
            )
            insert = text(
                f'insert into public."{table}" ({", ".join(columns)}) values ({placeholders})'
            )
            for row in rows:
                if table == "citizen_users":
                    upload_profile_picture(row)
                if table == "ctc_issuance":
                    upload_pdf(row)
                conn.execute(insert, {c: convert(row[c], types[c]) for c in columns})

            conn.execute(text(
                f"select setval(pg_get_serial_sequence('public.{table}', 'id'), "
                f"coalesce((select max(id) from public.\"{table}\"), 1), "
                f"(select max(id) from public.\"{table}\") is not null)"
            ))

    src.close()


def reupload_local_pdfs(engine) -> None:
    if not storage_service.is_enabled():
        sys.exit("SUPABASE_URL / SUPABASE_SECRET_KEY are not set.")
    with engine.begin() as conn:
        rows = conn.execute(text(
            "select id, ctc_number, pdf_path from public.ctc_issuance "
            "where pdf_path is not null and pdf_path not like 'supabase://%'"
        )).mappings().all()
        print(f"  {len(rows)} issuance(s) with a local PDF path")
        for r in rows:
            row = dict(r)
            upload_pdf(row)
            if row["pdf_path"] != r["pdf_path"]:
                conn.execute(
                    text("update public.ctc_issuance set pdf_path = :p where id = :id"),
                    {"p": row["pdf_path"], "id": row["id"]},
                )


def compare_counts(source: str, engine) -> bool:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    ok = True
    print(f"\n  {'table':<20}{'sqlite':>8}{'supabase':>10}")
    with engine.connect() as conn:
        for table in TABLES:
            local = src.execute(f'select count(*) from "{table}"').fetchone()[0]
            remote = conn.execute(text(f'select count(*) from public."{table}"')).scalar()
            mark = "" if local == remote else "  <-- MISMATCH"
            ok &= local == remote
            print(f"  {table:<20}{local:>8}{remote:>10}{mark}")
    src.close()
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="cedula.db", help="SQLite file to copy from")
    parser.add_argument("--pdfs-only", action="store_true", help="only re-upload local PDFs already in Supabase rows")
    args = parser.parse_args()

    if not settings.DATABASE_URL.startswith("postgresql"):
        sys.exit("DATABASE_URL in .env must point at Supabase (postgresql://...).")
    engine = create_engine(settings.DATABASE_URL, connect_args={"connect_timeout": 10})

    if args.pdfs_only:
        reupload_local_pdfs(engine)
        return

    if not os.path.isfile(args.source):
        sys.exit(f"SQLite file not found: {args.source}")
    print(f"Backup: {backup_sqlite(args.source)}")
    print(f"Storage uploads: {'enabled' if storage_service.is_enabled() else 'DISABLED (files stay inline/local)'}")
    print("Copying rows:")
    migrate(args.source, engine)
    print("Row counts:")
    sys.exit(0 if compare_counts(args.source, engine) else 1)


if __name__ == "__main__":
    main()
