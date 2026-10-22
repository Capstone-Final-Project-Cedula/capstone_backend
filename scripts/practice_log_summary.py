"""
Summary of the TEMPORARY practice session log (plan 001.7; plan 003 replaces it).

    python scripts/practice_log_summary.py --since-plan 001.7          (from capstone_backend, venv active)
    python scripts/practice_log_summary.py --since 2026-10-21 --staff ralph
    python scripts/practice_log_summary.py --since-plan 001.7 --include-real

Read-only: SELECTs through the backend's own database settings (.env DATABASE_URL), in a read-only
transaction on Postgres. Default: test sessions only (--include-real adds real-mode ones), voice sessions only.
"Since plan NNN" = sessions whose build label (PRACTICE_LOG_BUILD, e.g. "version-2.2+001.7") names plan NNN or a later one.

Writes capstone_docs/practice-log/<date>-since-<plan or date>-summary.md and .json. That folder is git-ignored:
it holds what Whisper heard and names typed during practice. Never writes to eval/results/.

Practice data from live use. Not an accuracy measurement; never used in the paper.
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # `python scripts/...` has only scripts/ on the path

OUT_DIR = ROOT.parent / "capstone_docs" / "practice-log"
DISCLAIMER = "Practice data from live use. Not an accuracy measurement; never used in the paper."
TOP_PAIRS = 5

# Same order as the interview.
FIELD_ORDER = [
    "surname", "given_name", "middle_name", "address_town", "address_barangay", "address_purok", "address",
    "date", "sex", "civil_status", "citizenship", "icr_no", "place_of_birth", "height", "weight",
    "occupation", "gross_annual_income",
]
# "Heard -> saved" pairs are only meaningful for words: a spoken date or number is never written the way it is said.
PAIR_FIELDS = {
    "surname", "given_name", "middle_name", "address_town", "address_barangay", "address_purok",
    "place_of_birth", "occupation", "citizenship",
}
# The joined address is a summary row: it has no tries or seconds of its own.
SUMMARY_FIELDS = {"address"}
VOICE = ("whisper", "browser")
# Why a turn came back empty (plan 002.2). Older builds logged only "no-speech".
EMPTY_REASONS = ("browser-no-speech", "too-short", "whisper-empty", "dropped-echo", "dropped-repetition",
                 "no-speech")
# How a question ended (plan 002.2). Skipped questions count as misses until finished; typing is never voice.
RESOLUTIONS = ("first_try", "ladder", "typed", "end_step_voice", "end_step_typed", "end_step_blank", "skipped")


def plan_tuple(plan: str):
    """'001.6.1' -> (1, 6, 1); None when it is not a plan number."""
    if not re.fullmatch(r"\d+(\.\d+)*", plan or ""):
        return None
    return tuple(int(part) for part in plan.split("."))


def build_plan(build_label):
    """The plan named after the '+' in a build label ('version-2.2+001.7' -> (1, 7)), or None."""
    if not build_label or "+" not in build_label:
        return None
    return plan_tuple(build_label.rsplit("+", 1)[1].strip())


def load_sessions(db, *, since_plan=None, since_date=None, staff=None, include_real=False):
    """Matching sessions as plain dicts with their answers. Only SELECTs."""
    from sqlalchemy import select
    from app.models.practice_log import PracticeLogAnswer, PracticeLogSession
    from app.models.staff_user import StaffUser

    query = (
        select(PracticeLogSession, StaffUser.username)
        .outerjoin(StaffUser, StaffUser.id == PracticeLogSession.staff_user_id)
        .where(PracticeLogSession.input_method == "voice")
        .order_by(PracticeLogSession.started_at, PracticeLogSession.id)
    )
    if not include_real:
        query = query.where(PracticeLogSession.mode == "test")
    if staff:
        query = query.where(StaffUser.username == staff)
    if since_date:
        query = query.where(PracticeLogSession.started_at >= datetime.combine(since_date, datetime.min.time()))

    sessions = []
    for session, username in db.execute(query).all():
        if since_plan is not None:
            plan = build_plan(session.build_label)
            if plan is None or plan < since_plan:
                continue
        answers = db.execute(
            select(PracticeLogAnswer).where(PracticeLogAnswer.session_id == session.id).order_by(PracticeLogAnswer.id)
        ).scalars().all()
        sessions.append({
            "id": session.id,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "build_label": session.build_label,
            "mode": session.mode,
            "staff": username,
            "outcome": session.outcome,
            "answers": [{
                "field": a.field,
                "attempts": a.attempts,
                "attempt_details": a.attempt_details or [],
                "first_value": a.first_value,
                "accepted_value": a.accepted_value,
                "accepted_engine": a.accepted_engine,
                "seconds": a.seconds,
                "final_value": a.final_value,
                "changed_after_voice": a.changed_after_voice,
            } for a in answers],
        })
    return sessions


def _first_heard(answer):
    """What the engine heard on the first try (typed or spelled tries are not 'heard')."""
    for attempt in answer["attempt_details"]:
        if attempt.get("engine") in ("whisper", "browser") and (attempt.get("heard") or "").strip():
            return attempt["heard"].strip()
    return None


def resolution(answer):
    """
    first_try: right by voice on the only try; ladder: by voice or spelling, later in the interview;
    typed: typed during the interview (staff pressed type); end_step_*: skipped, then said, typed or left
    blank at the end; skipped: never finished. Rows from before 002.2 have no stages: never end_step.
    """
    engine = answer["accepted_engine"]
    if engine == "skipped":
        return "skipped"
    tries = answer["attempt_details"]
    end = [a for a in tries if a.get("stage") == "end_step"]
    if end:
        if end[-1].get("reason") == "left-blank":
            return "end_step_blank"
        return "end_step_typed" if engine == "typed" else "end_step_voice"
    if engine == "typed":
        return "typed"
    if engine in VOICE and answer["attempts"] == 1:
        return "first_try"
    return "ladder"


def _rate(count, total):
    return round(count / total, 3) if total else None


def summarize(sessions):
    """Per-field numbers and the most common 'heard -> final' pairs, from load_sessions() output."""
    by_field = defaultdict(list)
    for session in sessions:
        for answer in session["answers"]:
            by_field[answer["field"]].append(answer)

    fields = []
    for field in sorted(by_field, key=lambda f: FIELD_ORDER.index(f) if f in FIELD_ORDER else len(FIELD_ORDER)):
        rows = by_field[field]
        known = [r for r in rows if r["changed_after_voice"] is not None]
        questions = [r for r in rows if field not in SUMMARY_FIELDS]
        seconds = [r["seconds"] for r in questions if r["seconds"] is not None]
        pairs = Counter()
        for r in rows if field in PAIR_FIELDS else []:
            heard = _first_heard(r)
            final = (r["final_value"] if r["final_value"] is not None else r["accepted_value"]) or ""
            # Only a real mis-hearing: not the same word, and not just a longer form ("Baungon" -> "Baungon, Bukidnon").
            if heard and final and heard.lower() not in final.lower():
                pairs[(heard, final)] += 1
        ends = Counter(resolution(r) for r in questions)
        empty = Counter(a.get("reason") for r in questions for a in r["attempt_details"]
                        if a.get("reason") in EMPTY_REASONS)
        fields.append({
            "field": field,
            "answers": len(rows),
            "changed_known": len(known),
            "changed_rate": round(sum(1 for r in known if r["changed_after_voice"]) / len(known), 3) if known else None,
            "average_tries": round(statistics.mean(r["attempts"] for r in questions), 2) if questions else None,
            # Plan 002.2's three numbers: first try, after the ladder (voice or spelling), finished at the end.
            "first_try_rate": _rate(ends["first_try"], len(questions)),
            "after_ladder_rate": _rate(ends["first_try"] + ends["ladder"], len(questions)),
            "end_step": {"said": ends["end_step_voice"], "typed": ends["end_step_typed"],
                         "blank": ends["end_step_blank"]},
            "skipped": sum(ends[k] for k in ("end_step_voice", "end_step_typed", "end_step_blank", "skipped")),
            "still_skipped": ends["skipped"],
            "typed_in_interview": ends["typed"],
            "empty_turns": dict(empty),
            "seconds_median": round(statistics.median(seconds), 1) if seconds else None,
            "seconds_max": round(max(seconds), 1) if seconds else None,
            "engines": dict(Counter(r["accepted_engine"] for r in rows)),
            "common_mishearings": [
                {"heard": heard, "final": final, "count": count}
                for (heard, final), count in pairs.most_common(TOP_PAIRS)
            ],
        })

    return {
        "sessions_read": len(sessions),
        "answers_read": sum(len(s["answers"]) for s in sessions),
        "outcomes": dict(Counter(s["outcome"] for s in sessions)),
        "builds": dict(Counter(s["build_label"] or "(none)" for s in sessions)),
        "skipped_questions": sum(f["skipped"] for f in fields),
        "fields": fields,
    }


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value):
    return "n/a" if value is None else f"{value}"


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary, filters):
    lines = [
        "# Practice log summary",
        f"_{DISCLAIMER}_",
        "",
        f"**Filters:** {filters}",
        f"**Sessions read:** {summary['sessions_read']} · **answers:** {summary['answers_read']} · "
        f"**outcomes:** {', '.join(f'{k} {v}' for k, v in sorted(summary['outcomes'].items())) or 'none'}",
        f"**Builds:** {', '.join(f'{k} ({v})' for k, v in sorted(summary['builds'].items())) or 'none'}",
        "",
    ]
    if not summary["fields"]:
        lines.append("No answers matched these filters.")
        return "\n".join(lines) + "\n"
    lines += [
        "## Per field",
        "",
        "| Field | Answers | Changed after voice | Avg tries | First-try | Seconds (median / max) | Engines |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in summary["fields"]:
        changed = f"{_pct(f['changed_rate'])} of {f['changed_known']}" if f["changed_known"] else "n/a (not issued)"
        engines = ", ".join(f"{k} {v}" for k, v in sorted(f["engines"].items()))
        lines.append(
            f"| {f['field']} | {f['answers']} | {changed} | {_num(f['average_tries'])} | {_pct(f['first_try_rate'])} | "
            f"{_num(f['seconds_median'])} / {_num(f['seconds_max'])} | {engines} |"
        )
    lines += [
        "",
        f"## The voice ladder (plan 002.2) · skipped questions: {summary.get('skipped_questions', 0)}",
        "_Skipped questions count as misses until finished; a typed answer is never a voice success._",
        "",
        "| Field | First try | After the ladder | Finished at the end (said / typed / blank) | Still skipped | "
        "Typed mid-interview | Empty turns |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in summary["fields"]:
        if f["field"] in SUMMARY_FIELDS:
            continue
        end = f["end_step"]
        empty = ", ".join(f"{k} {v}" for k, v in sorted(f["empty_turns"].items())) or "none"
        lines.append(
            f"| {f['field']} | {_pct(f['first_try_rate'])} | {_pct(f['after_ladder_rate'])} | "
            f"{end['said']} / {end['typed']} / {end['blank']} | {f['still_skipped']} | {f['typed_in_interview']} | "
            f"{empty} |"
        )
    lines += ["", "## Common mis-hearings (heard on the first try -> saved)", ""]
    any_pairs = False
    for f in summary["fields"]:
        if not f["common_mishearings"]:
            continue
        any_pairs = True
        lines.append(f"**{f['field']}**")
        for pair in f["common_mishearings"]:
            lines.append(f"- {_cell(pair['heard'])} -> {_cell(pair['final'])} (x{pair['count']})")
        lines.append("")
    if not any_pairs:
        lines.append("None: every first-try hearing matched what was saved.")
    return "\n".join(lines).rstrip() + "\n"


def output_paths(label, today=None):
    stamp = (today or date.today()).isoformat()
    slug = re.sub(r"[^0-9A-Za-z.-]+", "-", label)
    return OUT_DIR / f"{stamp}-since-{slug}-summary.md", OUT_DIR / f"{stamp}-since-{slug}-summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the temporary practice session log (read-only).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--since-plan", metavar="NNN", help="sessions built by this plan or later, e.g. 001.7")
    group.add_argument("--since", metavar="YYYY-MM-DD", help="sessions started on or after this date")
    parser.add_argument("--staff", metavar="USERNAME", help="only this staff account's sessions")
    parser.add_argument("--include-real", action="store_true", help="also real-mode sessions (default: test only)")
    args = parser.parse_args(argv)

    since_plan = since_date = None
    if args.since_plan:
        since_plan = plan_tuple(args.since_plan)
        if since_plan is None:
            parser.error("--since-plan must look like 001.7")
        label = args.since_plan
    else:
        try:
            since_date = date.fromisoformat(args.since)
        except ValueError:
            parser.error("--since must look like 2026-10-21")
        label = args.since

    from sqlalchemy import text
    from app.core.database import IS_POSTGRES, SessionLocal

    db = SessionLocal()
    try:
        if IS_POSTGRES:
            db.execute(text("SET TRANSACTION READ ONLY"))
        sessions = load_sessions(
            db, since_plan=since_plan, since_date=since_date, staff=args.staff, include_real=args.include_real,
        )
    finally:
        db.rollback()
        db.close()

    summary = summarize(sessions)
    filters = (
        f"{'plan ' + args.since_plan + ' or later' if args.since_plan else 'from ' + args.since}"
        f"{', staff ' + args.staff if args.staff else ''}"
        f"{', test and real sessions' if args.include_real else ', test sessions only'}, voice sessions"
    )
    md_path, json_path = output_paths(label)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(summary, filters), encoding="utf-8")
    json_path.write_text(
        json.dumps({"disclaimer": DISCLAIMER, "filters": filters, **summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Read {summary['sessions_read']} session(s), {summary['answers_read']} answer(s).")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
