"""
Service execution module: Lead Research Report
================================================
This is the first "fulfilment_module" referenced in the services table.
Given a job spec (what the customer agreed to), it queries the business
database and produces the deliverable file(s).

Platform-shaped: this file knows nothing about customers, conversations,
or payments. It takes a spec dict in, produces files out. That's the
contract every future service module (SEO audit, automation script, etc)
will also follow — so System 1-2-4 never need to know HOW a service works,
only that they can call `fulfil(spec) -> list[file_path]`.

Usage:
    spec = {
        "category": "dental_clinic",       # must match a category in the db, or a list of them
        "country": "US",                   # ISO2, optional
        "region": "TX",                    # optional, matches 'region' column
        "no_website": True,                # filter to only businesses with no website
        "count": 300,                      # how many rows the client is paying for
        "output_formats": ["csv", "pdf"],  # which files to produce
    }
    files = fulfil(spec, db_path="business_database.parquet", out_dir="./deliverables/job123")
"""

import os
import re
import json
import duckdb
import pandas as pd
from datetime import datetime, timezone


class SpecError(ValueError):
    """Raised when the job spec can't be fulfilled as written — never guess, always surface it."""


# Country calling codes for the countries this has actually been tested against.
# Add more as real jobs need them — deliberately small so an unmapped country
# fails loudly (see _normalize_phone) instead of silently guessing a wrong code.
_COUNTRY_CALLING_CODE = {
    "US": "1", "CA": "1", "GB": "44", "NG": "234", "IN": "91",
    "DE": "49", "FR": "33", "AU": "61", "BR": "55",
}


def _normalize_phone(raw, country: str | None) -> str:
    """Raw Overture phone data mixes '+16142921472' and '16142921472' and
    '6142921472' for the same country, which looks unprofessional in a
    deliverable and breaks click-to-call. Standardize to '+<code><digits>'.
    Returns the digits-only value unchanged (with a note) if the country's
    calling code isn't in the map above — never guesses one."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
        return ""
    digits = re.sub(r"\D", "", str(raw))
    if not digits:
        return ""
    if not country or country not in _COUNTRY_CALLING_CODE:
        return digits  # unknown calling code — return cleaned digits rather than a wrong +prefix
    code = _COUNTRY_CALLING_CODE[country]
    if digits.startswith(code) and len(digits) > len(code):
        return f"+{digits}"
    return f"+{code}{digits}"


def _build_where(spec: dict, cols: set) -> tuple[str, list]:
    clauses, params = [], []

    cat = spec.get("category")
    if cat:
        cats = [cat] if isinstance(cat, str) else list(cat)
        clauses.append(f"category IN ({','.join(['?'] * len(cats))})")
        params.extend(cats)

    if spec.get("country"):
        clauses.append("country = ?")
        params.append(spec["country"])

    if spec.get("region"):
        clauses.append("region = ?")
        params.append(spec["region"])

    if spec.get("city"):
        clauses.append("city = ?")
        params.append(spec["city"])

    if spec.get("no_website") is True:
        clauses.append("(website IS NULL OR trim(website) = '')")
    elif spec.get("no_website") is False:
        clauses.append("website IS NOT NULL AND trim(website) != ''")

    if spec.get("has_phone") is True:
        clauses.append("(phone IS NOT NULL AND trim(phone) != '')")

    if spec.get("has_email") is True:
        clauses.append("(email IS NOT NULL AND trim(email) != '')")

    if not clauses:
        raise SpecError("Spec has no filters at all — refusing to export the entire database.")

    return " AND ".join(clauses), params


def preview_count(spec: dict, db_path: str) -> int:
    """Always call this BEFORE selling the job. If it returns fewer rows than the
    customer is paying for, that's a real constraint to renegotiate on, not something
    to pad with fabricated rows."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW biz AS SELECT * FROM read_parquet('{db_path}')")
    cols = {r[0] for r in con.execute("DESCRIBE biz").fetchall()}
    where, params = _build_where(spec, cols)
    n = con.execute(f"SELECT count(*) FROM biz WHERE {where}", params).fetchone()[0]
    con.close()
    return n


def fulfil(spec: dict, db_path: str, out_dir: str) -> list:
    """Produces the deliverable file(s) for a lead-research-report job.
    Returns a list of file paths actually written — never more than were agreed."""
    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW biz AS SELECT * FROM read_parquet('{db_path}')")
    cols = {r[0] for r in con.execute("DESCRIBE biz").fetchall()}
    where, params = _build_where(spec, cols)

    available = con.execute(f"SELECT count(*) FROM biz WHERE {where}", params).fetchone()[0]
    requested = spec.get("count")
    if requested and available < requested:
        raise SpecError(
            f"Only {available:,} matching businesses exist for this spec; "
            f"{requested:,} were promised. Fix the spec or renegotiate scope before fulfilling."
        )

    limit = requested or available
    cols_to_select = "name, category, address, city, region, postcode, country, phone, website, email, social, brand"
    df = con.execute(
        f"SELECT {cols_to_select} FROM biz WHERE {where} ORDER BY random() LIMIT {limit}",
        params,
    ).df()
    con.close()

    if df.empty:
        raise SpecError("Query matched zero rows — spec is too narrow or wrong. Do not deliver an empty file.")

    df["phone"] = df["phone"].apply(lambda p: _normalize_phone(p, spec.get("country")))

    files = []
    formats = spec.get("output_formats", ["csv"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    if "csv" in formats:
        path = os.path.join(out_dir, f"lead_report_{stamp}.csv")
        df.to_csv(path, index=False)
        files.append(path)

    if "xlsx" in formats:
        path = os.path.join(out_dir, f"lead_report_{stamp}.xlsx")
        df.to_excel(path, index=False)
        files.append(path)

    if "pdf" in formats:
        path = os.path.join(out_dir, f"lead_report_{stamp}.pdf")
        _write_pdf_summary(df, spec, path)
        files.append(path)

    manifest = {
        "spec": spec,
        "rows_delivered": len(df),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    files.append(manifest_path)

    return files


def _write_pdf_summary(df: pd.DataFrame, spec: dict, path: str):
    """A short cover-page PDF summarizing the report, not the raw 300-row dump —
    the CSV/XLSX is the working file, this is what a client opens first."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = [
        Paragraph("Lead Research Report", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Rows delivered: {len(df):,}", styles["Normal"]),
        Paragraph(f"Filters: {json.dumps({k: v for k, v in spec.items() if k != 'output_formats'})}", styles["Normal"]),
        Spacer(1, 0.3 * inch),
    ]

    by_cat = df["category"].value_counts().head(15)
    rows = [["Category", "Count"]] + [[c, str(n)] for c, n in by_cat.items()]
    t = Table(rows, colWidths=[3 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    doc.build(story)


def run_smoke_test(db_path: str):
    """Call this yourself, with the real path to business_database.parquet, e.g.:
        run_smoke_test("/kaggle/input/global-businesses-v1/business_database.parquet")
    This is NOT run automatically — notebooks (Jupyter/Kaggle/Colab) always satisfy
    `if __name__ == "__main__"` and hijack sys.argv for their own kernel internals,
    so auto-running on import/paste is unsafe in that environment. Call it explicitly instead."""
    spec = {
        "category": "dental_clinic",
        "country": "US",
        "no_website": True,
        "count": 50,
        "output_formats": ["csv", "pdf"],
    }
    n = preview_count(spec, db_path)
    print(f"Preview: {n:,} matching rows available for this spec")
    if n >= spec["count"]:
        files = fulfil(spec, db_path, "./test_deliverable")
        print("Produced:", files)
    else:
        print(f"Not enough rows ({n}) to fulfil a {spec['count']}-row job — would renegotiate scope here.")


if __name__ == "__main__":
    # Only runs when this file is executed as a real script from a terminal
    # (python service_lead_research_report.py path/to.parquet) — never in a notebook,
    # since notebooks always set __name__ == "__main__" too and would hit this by accident.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "-f":
        run_smoke_test(sys.argv[1])
    else:
        print("No real file path given on the command line — skipping the smoke test.")
        print('In a notebook, call run_smoke_test("/path/to/business_database.parquet") yourself instead.')
