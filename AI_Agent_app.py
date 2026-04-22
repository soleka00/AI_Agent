import os
import re
import json
import unicodedata
import ast
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import base64
import tempfile
from io import BytesIO
import html

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# Optional: used for Mermaid -> PNG in PDF 
import requests

# ============================================================
# Robust JSON parsing helpers (LLM outputs are messy)
# ============================================================

SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u00ab": '"', "\u00bb": '"',
}

def _basic_json_cleanup(text: str) -> str:
    for k, v in SMART_QUOTES.items():
        text = text.replace(k, v)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text

def extract_first_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Model output is not a string.")
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = _basic_json_cleanup(s)

    start = s.find("{")
    if start < 0:
        raise ValueError("No JSON object found (missing '{').")
    s2 = s[start:]

    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(s2):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    if end is None:
        last = s2.rfind("}")
        if last > 0:
            candidate = s2[: last + 1]
        else:
            raise ValueError("JSON object seems unterminated (no matching '}').")
    else:
        candidate = s2[:end]

    candidate = _basic_json_cleanup(candidate)
    return json.loads(candidate)

def parse_llm_json_or_repair(
    client: Optional[InferenceClient],
    raw_text: str,
    *,
    max_tokens: int = 900,
    max_repairs: int = 2,
) -> Dict[str, Any]:
    try:
        return extract_first_json_object(raw_text)
    except Exception as e0:
        last_err = e0

    if client is None:
        return {"_parse_error": f"{type(last_err).__name__}: {last_err}", "_raw": (raw_text or "")[:8000]}

    system = (
        "You are a strict JSON repair tool. "
        "Return ONLY valid JSON (a single JSON object). "
        "No commentary. No markdown. No trailing commas. "
        "Use double quotes for all keys and string values."
    )

    text_to_fix = raw_text
    for _ in range(max(0, int(max_repairs)) + 1):
        user = (
            "Convert the following content into ONE valid JSON object.\n"
            "- Remove any non-JSON text.\n"
            "- Ensure strings are properly closed.\n"
            "- Ensure all keys are quoted.\n"
            "- Ensure there are no trailing commas.\n\n"
            f"{text_to_fix}"
        )
        try:
            resp = client.chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
            repaired = resp.choices[0].message.content
            try:
                return extract_first_json_object(repaired)
            except Exception as e1:
                last_err = e1
                text_to_fix = repaired
        except Exception as e2:
            last_err = e2
            break

    return {"_parse_error": f"{type(last_err).__name__}: {last_err}", "_raw": (raw_text or "")[:8000]}

# ============================================================
# Mermaid rendering (Streamlit UI)
# ============================================================

def render_mermaid(mermaid_code: str, height: int = 650):
    """Render Mermaid diagram in Streamlit."""
    html_block = (
        '<div class="mermaid" style="background-color:white;padding:16px;border-radius:12px;border:1px solid #ddd;">'
        + mermaid_code +
        '</div>'
        '<script type="module">'
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
        'mermaid.initialize({startOnLoad:true, theme:"default", securityLevel:"loose", er:{useMaxWidth:true}});'
        '</script>'
    )
    components.html(html_block, height=height, scrolling=True)

# ============================================================
# AUDIT LOGGING & REPORTING
# ============================================================

def audit_init():
    """Initialize audit session state."""
    if "audit" not in st.session_state:
        st.session_state.audit = {
            "started_at": datetime.utcnow().isoformat() + "Z",
            "task_text": None,
            "events": [],
        }

def audit_event(mode: str, step: str, data: Any = None):
    """Append timestamped audit event."""
    audit_init()
    st.session_state.audit["events"].append({
        "ts": datetime.utcnow().isoformat() + "Z",
        "mode": str(mode),
        "step": str(step),
        "data": data if data is not None else {},
    })

def audit_set_task(task_text: str):
    """Record initial business task."""
    audit_init()
    st.session_state.audit["task_text"] = task_text
    audit_event("global", "task_set", {"task_text": task_text})

def audit_snapshot() -> Dict[str, Any]:
    """Return current audit state as a snapshot."""
    audit_init()
    return {
        "started_at": st.session_state.audit.get("started_at"),
        "task_text": st.session_state.audit.get("task_text"),
        "events": list(st.session_state.audit.get("events", [])),
    }

def _safe_str(x: Any, max_len: int = 220) -> str:
    """Safely convert to string with length limit."""
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= max_len else (s[:max_len - 1] + "…")

# ============================================================
# AUDIT EXTRACTION HELPERS
# ============================================================

def _extract_task_text(audit: Dict[str, Any]) -> Optional[str]:
    """Find business task from audit."""
    if audit.get("task_text"):
        return audit["task_text"]
    for ev in audit.get("events", []):
        if ev.get("step") == "task_set":
            d = ev.get("data") or {}
            if isinstance(d, dict) and d.get("task_text"):
                return d["task_text"]
    return None

def _extract_er_agent_version(audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract initial LLM-generated ER brief."""
    for ev in audit.get("events", []):
        if ev.get("mode") == "business" and ev.get("step") == "extract_er":
            data = ev.get("data") or {}
            if isinstance(data, dict):
                return data
    return None

def _extract_er_approved_version(audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract approved ER brief (after HITL Step 2)."""
    for ev in audit.get("events", []):
        if ev.get("mode") == "business" and ev.get("step") == "er_approved":
            data = ev.get("data") or {}
            if isinstance(data, dict):
                return data
    return None

def _extract_schema_raw(audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract raw schema from LLM (Step 3)."""
    for ev in audit.get("events", []):
        if ev.get("mode") == "business" and ev.get("step") == "schema_raw":
            data = ev.get("data") or {}
            if isinstance(data, dict):
                return data
    return None

def _find_mermaid_code(audit: Dict[str, Any]) -> Optional[str]:
    """Find Mermaid ERD code in audit events."""
    for ev in reversed(audit.get("events", [])):
        d = ev.get("data")
        if not isinstance(d, dict):
            continue
        mermaid = d.get("mermaid") or d.get("mermaid_code")
        if isinstance(mermaid, str) and mermaid.strip().startswith(("erDiagram", "flowchart", "graph")):
            return mermaid.strip()
    return None

def _find_sql_ddl(audit: Dict[str, Any]) -> Optional[str]:
    """Find SQL DDL code in audit events."""
    for ev in reversed(audit.get("events", [])):
        d = ev.get("data")
        if not isinstance(d, dict):
            continue
        sql = d.get("ddl") or d.get("sql_ddl") or d.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql.strip()
    return None

# ============================================================
# PDF REPORT BUILDER (fixes Czech diacritics + embeds Mermaid as image)
# ============================================================

def _fallback_text_report(audit: Dict[str, Any], reason: str) -> bytes:
    txt = {
        "error": reason,
        "audit": audit,
    }
    return json.dumps(txt, ensure_ascii=False, indent=2).encode("utf-8")

def _strip_para_wrappers(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    s = txt.strip()
    s = re.sub(r"^\s*<\s*para\s*>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*<\s*/\s*para\s*>\s*$", "", s, flags=re.IGNORECASE)
    return s

def _escape_for_reportlab_paragraph(txt: str) -> str:
    s = "" if txt is None else str(txt)
    s = _strip_para_wrappers(s)
    s = html.escape(s, quote=False)
    s = s.replace("\n", "<br/>")
    return s

def _find_first_existing_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    return None

def register_unicode_font() -> Tuple[str, str, str]:
    """
    Returns (regular_font_name, bold_font_name, mono_font_name).
    Uses local TTF files from the repo to guarantee Czech diacritics in PDFs.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Courier"

    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

    # EXPECTED PATHS INSIDE YOUR HF SPACE REPO (put fonts into src/)
    regular_path = os.path.join(base_dir, "NotoSans-Regular.ttf")
    bold_path = os.path.join(base_dir, "NotoSans-Bold.ttf")
    mono_path = os.path.join(base_dir, "NotoSansMono-Regular.ttf")  # optional

    # Hard failover if the font files are missing
    if not os.path.isfile(regular_path):
        return "Helvetica", "Helvetica-Bold", "Courier"

    regular_name = "CZFont"
    bold_name = "CZFontBold"
    mono_name = "CZMono"

    try:
        pdfmetrics.registerFont(TTFont(regular_name, regular_path))
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Courier"

    # Bold (optional, but recommended)
    if os.path.isfile(bold_path):
        try:
            pdfmetrics.registerFont(TTFont(bold_name, bold_path))
        except Exception:
            bold_name = regular_name
    else:
        bold_name = regular_name

    # Mono (optional; if missing, we’ll reuse regular)
    if os.path.isfile(mono_path):
        try:
            pdfmetrics.registerFont(TTFont(mono_name, mono_path))
        except Exception:
            mono_name = regular_name
    else:
        mono_name = regular_name

    return regular_name, bold_name, mono_name

def mermaid_to_png_bytes_via_mermaid_ink(mermaid_code: str, timeout_s: int = 12) -> Optional[bytes]:
    """
    Convert Mermaid source -> PNG using mermaid.ink (requires internet).
    If unreachable, returns None and PDF will include Mermaid source text only.
    """
    if not isinstance(mermaid_code, str) or not mermaid_code.strip():
        return None
    code = mermaid_code.strip()
    b64 = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii").rstrip("=")
    url = f"https://mermaid.ink/img/{b64}"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        return None
    return None

def build_pdf_report_bytes(audit: Dict[str, Any]) -> bytes:
    """
    Generate a PDF report with UTF-8/Czech support and embedded Mermaid diagram (best-effort).
    """
    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, PageBreak, Preformatted
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from io import BytesIO as IO
    except ImportError:
        return _fallback_text_report(audit, "Missing reportlab library")

    try:
        font_name, font_bold, font_mono = register_unicode_font()

        styles = getSampleStyleSheet()
        h1 = ParagraphStyle(
            "h1",
            parent=styles["Heading1"],
            fontName=font_bold,
            fontSize=16,
            leading=20,
            spaceAfter=12,
            textColor=colors.HexColor("#003366"),
        )
        h2 = ParagraphStyle(
            "h2",
            parent=styles["Heading2"],
            fontName=font_bold,
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor("#005599"),
        )
        h3 = ParagraphStyle(
            "h3",
            parent=styles["Heading3"],
            fontName=font_bold,
            fontSize=11,
            leading=13,
            spaceBefore=6,
            spaceAfter=6,
        )
        body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font_name, fontSize=9, leading=12)
        small = ParagraphStyle("small", parent=styles["BodyText"], fontName=font_name, fontSize=8, leading=10)
        pre = ParagraphStyle("pre", parent=styles["Code"], fontName=font_mono, fontSize=8, leading=10)

        def P(txt: str, style=body) -> Paragraph:
            safe = _escape_for_reportlab_paragraph(txt)
            return Paragraph(safe, style)

        def _make_table(rows: List[List[Any]], col_widths=None) -> Table:
            safe_rows = []
            for i, row in enumerate(rows):
                safe_row = []
                for cell in row:
                    if isinstance(cell, Paragraph):
                        safe_row.append(cell)
                    else:
                        s = _safe_str(cell) if cell else ""
                        cell_style = small if i == 0 else body
                        safe_row.append(Paragraph(_escape_for_reportlab_paragraph(s), cell_style))
                safe_rows.append(safe_row)

            t = Table(safe_rows, colWidths=col_widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ]))
            return t

        story = []

        # PAGE 1: COVER
        story.append(P("AI Data Modeling Agent", style=h1))
        story.append(P("Comprehensive Audit Report", style=h2))
        story.append(Spacer(1, 30))
        story.append(P(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", style=small))
        story.append(P(f"<b>Session started:</b> {_safe_str(audit.get('started_at'))}", style=small))
        story.append(Spacer(1, 30))

        # Summary metrics
        er_agent = _extract_er_agent_version(audit)
        er_approved = _extract_er_approved_version(audit)
        schema_raw = _extract_schema_raw(audit)

        story.append(P("Summary", style=h3))
        summary_rows = [
            ["Metric", "Value"],
            ["Audit events", str(len(audit.get("events", [])))],
            ["ER entities (agent)", str(len(er_agent.get("entities", []))) if er_agent else "0"],
            ["ER approved", "✅ Yes" if er_approved else "⬜ No"],
            ["Schema tables (agent)", str(len(schema_raw.get("suggested_tables", []))) if schema_raw else "0"],
        ]
        story.append(_make_table(summary_rows, col_widths=[250, 150]))
        story.append(PageBreak())

        # PAGE 2: BUSINESS TASK
        story.append(P("1. Business Assignment", style=h2))
        task = _extract_task_text(audit)
        if task:
            story.append(P(task, style=body))
        else:
            story.append(P("(No task text found)", style=body))
        story.append(Spacer(1, 20))
        story.append(PageBreak())

        # PAGE 3-4: ER AGENT SUGGESTION
        story.append(P("2. ER Conceptual Model (Agent Suggestion)", style=h2))
        if not er_agent:
            story.append(P("(No LLM-generated ER found)", style=body))
        else:
            ents = er_agent.get("entities", []) or []
            rels = er_agent.get("relationships", []) or []
            rules = er_agent.get("business_rules", []) or []
            assumptions = er_agent.get("assumptions", []) or []

            if ents:
                story.append(P("Entities", style=h3))
                rows = [["Entity", "Description", "Attributes"]]
                for e in ents[:20]:
                    if isinstance(e, dict):
                        attrs = ", ".join(e.get("attributes", []) or [])[:100]
                        rows.append([
                            e.get("name", ""),
                            _safe_str(e.get("description", ""))[:80],
                            attrs,
                        ])
                story.append(_make_table(rows, col_widths=[100, 180, 180]))
                story.append(Spacer(1, 10))

            if rels:
                story.append(P("Relationships", style=h3))
                rows = [["From", "To", "Cardinality", "Notes"]]
                for r in rels[:20]:
                    if isinstance(r, dict):
                        rows.append([
                            r.get("from", ""),
                            r.get("to", ""),
                            r.get("cardinality", ""),
                            (r.get("notes", "") or "")[:80],
                        ])
                story.append(_make_table(rows, col_widths=[80, 80, 120, 180]))
                story.append(Spacer(1, 10))

            if rules:
                story.append(P("Business Rules", style=h3))
                for rule in rules[:10]:
                    story.append(P(f"• {rule}", style=body))

            if assumptions:
                story.append(Spacer(1, 8))
                story.append(P("Assumptions", style=h3))
                for assumption in assumptions[:10]:
                    story.append(P(f"• {assumption}", style=body))

        story.append(PageBreak())

        # PAGE 5-6: ER APPROVED (HITL)
        story.append(P("3. ER Model (After HITL Review)", style=h2))
        if not er_approved:
            story.append(P("(No HITL approval recorded)", style=body))
        else:
            ents = er_approved.get("entities", []) or []
            rels = er_approved.get("relationships", []) or []

            if ents:
                story.append(P("Approved Entities", style=h3))
                rows = [["Entity", "Description", "Attributes"]]
                for e in ents[:20]:
                    if isinstance(e, dict):
                        attrs = ", ".join(e.get("attributes", []) or [])[:100]
                        rows.append([
                            e.get("name", ""),
                            _safe_str(e.get("description", ""))[:80],
                            attrs,
                        ])
                story.append(_make_table(rows, col_widths=[100, 180, 180]))
                story.append(Spacer(1, 10))

            if rels:
                story.append(P("Approved Relationships", style=h3))
                rows = [["From", "To", "Cardinality", "Notes"]]
                for r in rels[:20]:
                    if isinstance(r, dict):
                        rows.append([
                            r.get("from", ""),
                            r.get("to", ""),
                            r.get("cardinality", ""),
                            (r.get("notes", "") or "")[:80],
                        ])
                story.append(_make_table(rows, col_widths=[80, 80, 120, 180]))

        story.append(PageBreak())

        # PAGE 7-8: SCHEMA RAW (AGENT SUGGESTION)
        story.append(P("4. Relational Schema (Agent Suggestion)", style=h2))
        if not schema_raw:
            story.append(P("(No raw schema found)", style=body))
        else:
            tables = schema_raw.get("suggested_tables", []) or []
            if tables:
                story.append(P("Suggested Tables", style=h3))
                rows = [["Table", "Columns", "Constraints"]]
                for t in tables[:20]:
                    if isinstance(t, dict):
                        cols = ", ".join([str(c) for c in t.get("columns", [])[:6]])
                        cons_count = len(t.get("constraints", []) or [])
                        rows.append([t.get("table", ""), cols if cols else "(none)", str(cons_count)])
                story.append(_make_table(rows, col_widths=[100, 250, 100]))
                story.append(Spacer(1, 10))

            rels = schema_raw.get("relationships", []) or []
            if rels:
                story.append(P("Suggested Relationships", style=h3))
                rows = [["From", "To", "Cardinality", "FK"]]
                for r in rels[:20]:
                    if isinstance(r, dict):
                        rows.append([r.get("from", ""), r.get("to", ""), r.get("cardinality", ""), r.get("fk", "")])
                story.append(_make_table(rows, col_widths=[90, 90, 120, 140]))

        story.append(PageBreak())

        # PAGE 9-10: SCHEMA FINAL
        story.append(P("5. Relational Schema (Final - After HITL)", style=h2))

        schema_final = None
        for ev in reversed(audit.get("events", [])):
            if ev.get("mode") == "business" and ev.get("step") in ("schema_hitl_saved", "schema_postprocess_validate"):
                data = ev.get("data")
                if isinstance(data, dict) and data.get("suggested_tables"):
                    schema_final = data
                    break

        if not schema_final:
            story.append(P("(No final schema found - still in progress)", style=body))
        else:
            tables = schema_final.get("suggested_tables", []) or []
            if tables:
                story.append(P("Final Tables", style=h3))
                rows = [["Table", "Columns", "Constraints"]]
                for t in tables[:20]:
                    if isinstance(t, dict):
                        cols = ", ".join([str(c) for c in t.get("columns", [])[:6]])
                        cons_count = len(t.get("constraints", []) or [])
                        rows.append([t.get("table", ""), cols if cols else "(none)", str(cons_count)])
                story.append(_make_table(rows, col_widths=[100, 250, 100]))
                story.append(Spacer(1, 10))

            rels = schema_final.get("relationships", []) or []
            if rels:
                story.append(P("Final Relationships", style=h3))
                rows = [["From", "To", "Cardinality", "FK"]]
                for r in rels[:20]:
                    if isinstance(r, dict):
                        rows.append([r.get("from", ""), r.get("to", ""), r.get("cardinality", ""), r.get("fk", "")])
                story.append(_make_table(rows, col_widths=[90, 90, 120, 140]))

        story.append(PageBreak())

        # PAGE 11: DIAGRAM (embedded PNG if possible)
        story.append(P("6. Data Model Diagram (ERD)", style=h2))
        mermaid_code = _find_mermaid_code(audit)
        if mermaid_code:
            png = mermaid_to_png_bytes_via_mermaid_ink(mermaid_code)
            if png:
                story.append(P("Rendered diagram (Mermaid → PNG):", style=h3))
                img = RLImage(IO(png))
                img._restrictSize(17.0 * cm, 22.0 * cm)
                story.append(img)
                story.append(Spacer(1, 8))
            else:
                story.append(P("Diagram image render unavailable; including Mermaid source.", style=small))

            story.append(P("Mermaid ERD Source:", style=h3))
            story.append(Preformatted(mermaid_code, pre))
        else:
            story.append(P("(No diagram found in audit)", style=body))

        story.append(PageBreak())

        # PAGE 12: SQL DDL
        story.append(P("7. SQL DDL", style=h2))
        ddl_code = _find_sql_ddl(audit)
        if ddl_code:
            story.append(P("Generated SQL DDL:", style=h3))
            story.append(Preformatted(ddl_code, pre))
        else:
            story.append(P("(No SQL DDL found)", style=body))

        story.append(PageBreak())

        # PAGE 13: AUDIT TIMELINE
        story.append(P("8. Audit Timeline", style=h2))
        rows = [["Timestamp", "Mode", "Step", "Summary"]]
        for ev in audit.get("events", [])[-80:]:
            ts = (ev.get("ts", "") or "")[:19]
            mode = ev.get("mode", "")
            step = ev.get("step", "")
            data = ev.get("data", {})

            summary = ""
            if isinstance(data, dict):
                if "entities" in data:
                    summary = f"Entities: {len(data.get('entities', []))}"
                elif "suggested_tables" in data:
                    summary = f"Tables: {len(data.get('suggested_tables', []))}"
                else:
                    try:
                        summary = json.dumps(data, ensure_ascii=False)[:60]
                    except Exception:
                        summary = str(data)[:60]
            else:
                summary = str(data)[:60]

            rows.append([ts, mode, step, summary])

        story.append(_make_table(rows, col_widths=[110, 70, 120, 160]))

        buf = IO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )
        doc.build(story)
        return buf.getvalue()

    except Exception as e:
        return _fallback_text_report(audit, f"{type(e).__name__}: {e}")

# ============================================================
# Identifier helpers
# ============================================================

def sanitize_identifier(name: str) -> str:
    """Normalize identifiers to ASCII snake_case."""
    s = str(name or "").strip()
    if not s:
        return "unnamed"

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()

    s = re.sub(r"[\s\-\/\.]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        return "unnamed"
    if re.match(r"^\d", s):
        s = f"t_{s}"
    return s

def _safe_ident(x: Any) -> str:
    return sanitize_identifier(x)

def _title_ident(x: Any) -> str:
    return _safe_ident(x).upper()

# ============================================================
# Safe LLM call
# ============================================================

def safe_chat_completion(
    client: Optional[InferenceClient],
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Tuple[Optional[str], Optional[str]]:
    if client is None:
        return None, "NO_CLIENT"
    try:
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content, None
    except HfHubHTTPError as e:
        msg = str(e)
        if "402" in msg or "Payment Required" in msg or "Credit balance is depleted" in msg:
            st.session_state["llm_available"] = False
            return None, "HF_402"
        return None, "HF_HTTP"
    except Exception:
        return None, "LLM_ERROR"

# ============================================================
# Business modelling: ER brief -> schema blueprint
# ============================================================

FK_RE = re.compile(
    r"foreign\s+key\s*\(\s*([a-zA-Z0-9_]+)\s*\)\s*references\s*([a-zA-Z0-9_]+)\s*\(\s*([a-zA-Z0-9_]+)\s*\)",
    re.IGNORECASE,
)

CARD_CANON = {
    "1:n": "one-to-many",
    "one_to_many": "one-to-many",
    "one-to-many": "one-to-many",
    "n:1": "many-to-one",
    "many_to_one": "many-to-one",
    "many-to-one": "many-to-one",
    "m:n": "many-to-many",
    "many_to_many": "many-to-many",
    "many-to-many": "many-to-many",
    "1:1": "one-to-one",
    "one_to_one": "one-to-one",
    "one-to-one": "one-to-one",
}

def canon_cardinality(x: Any) -> str:
    s = str("" if x is None else x).strip().lower().replace(" ", "_")
    s = s.replace("__", "_")
    return CARD_CANON.get(s, s.replace("_", "-") or "one-to-many")

def normalize_er_brief(er: Dict[str, Any]) -> Dict[str, Any]:
    er = dict(er) if isinstance(er, dict) else {}
    ents = er.get("entities", [])
    rels = er.get("relationships", [])
    rules = er.get("business_rules", [])
    assumptions = er.get("assumptions", [])

    if not isinstance(ents, list):
        ents = []
    if not isinstance(rels, list):
        rels = []
    if not isinstance(rules, list):
        rules = []
    if not isinstance(assumptions, list):
        assumptions = []

    norm_ents = []
    for e in ents:
        if isinstance(e, str):
            norm_ents.append({"name": sanitize_identifier(e), "description": "", "attributes": []})
        elif isinstance(e, dict):
            name = sanitize_identifier(e.get("name") or e.get("entity") or "")
            if not name:
                continue
            attrs = e.get("attributes") or []
            if isinstance(attrs, str):
                attrs = [x.strip() for x in attrs.split(",") if x.strip()]
            if not isinstance(attrs, list):
                attrs = []
            attrs = [sanitize_identifier(a) for a in attrs if str(a).strip()]
            norm_ents.append({
                "name": name,
                "description": str(e.get("description") or "")[:400],
                "attributes": attrs,
            })

    norm_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        f = sanitize_identifier(r.get("from"))
        t = sanitize_identifier(r.get("to"))
        if not f or not t:
            continue
        norm_rels.append({
            "from": f,
            "to": t,
            "cardinality": canon_cardinality(r.get("cardinality", "one-to-many")),
            "fk_on": ("" if sanitize_identifier(r.get("fk_on") or "") in ("unnamed","none","null") else sanitize_identifier(r.get("fk_on") or "")),
            "fk_name": ("" if sanitize_identifier(r.get("fk_name") or "") in ("unnamed","none","null") else sanitize_identifier(r.get("fk_name") or "")),
            "notes": str(r.get("notes") or "")[:500],
        })

    er["entities"] = norm_ents
    er["relationships"] = norm_rels
    er["business_rules"] = [str(x).strip() for x in rules if str(x).strip()]
    er["assumptions"] = [str(x).strip() for x in assumptions if str(x).strip()]
    return er

def extract_er_brief_llm(client: Optional[InferenceClient], business_task: str) -> Dict[str, Any]:
    system = (
        "You are a senior Data Architect. Extract an ER modelling brief from the business text.\n"
        "Return ONLY valid JSON with keys:\n"
        "entities (list of objects), relationships (list of objects), business_rules (list), assumptions (list).\n\n"
        "entities item format:\n"
        "{ name: snake_case, description: string, attributes: [snake_case,...] }\n\n"
        "relationships item format:\n"
        "{ from: entity, to: entity, cardinality: one-to-one|one-to-many|many-to-one|many-to-many,\n"
        "  fk_on: entity(optional), fk_name: snake_case(optional), notes: string(optional) }\n\n"
        "Rules:\n"
        "- Use snake_case identifiers.\n"
        "- Do NOT invent implementation tables here; focus on conceptual entities and relationships.\n"
        "- If uncertain, state assumptions.\n"
        "No markdown, no commentary."
    )
    payload = {"business_task": business_task}
    raw, err = safe_chat_completion(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        max_tokens=1500,
    )
    if err or raw is None:
        return {"_llm_error": err or "UNKNOWN", "entities": [], "relationships": [], "business_rules": [], "assumptions": []}
    out = parse_llm_json_or_repair(client, raw, max_repairs=2)
    return normalize_er_brief(out)

def drop_invalid_tables(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp) if isinstance(bp, dict) else {}
    cleaned = []

    for t in bp.get("suggested_tables", []) or []:
        if not isinstance(t, dict):
            continue

        name = _safe_ident(t.get("table"))
        if not name or name in ("none", "null", "unnamed"):
            continue

        raw_cols = t.get("columns") or []
        cols: List[str] = []

        if isinstance(raw_cols, list):
            for c in raw_cols:
                if isinstance(c, dict):
                    c = c.get("name") or c.get("column") or c.get("field") or ""
                cols.append(_safe_ident(c))
        elif isinstance(raw_cols, dict):
            for k in raw_cols.keys():
                cols.append(_safe_ident(k))
        else:
            cols = [_safe_ident(x) for x in str(raw_cols).split(",") if str(x).strip()]

        cols = [c for c in cols if c and c not in ("unnamed", "none", "null")]
        cols = [c for c in cols if not c.startswith("name_id_type_")]

        if not cols:
            continue

        cleaned.append({
            "table": name,
            "columns": cols,
            "constraints": [str(x) for x in (t.get("constraints") or [])],
        })

    bp["suggested_tables"] = cleaned
    if not isinstance(bp.get("relationships"), list):
        bp["relationships"] = []
    if not isinstance(bp.get("assumptions"), list):
        bp["assumptions"] = []
    if not isinstance(bp.get("kpis"), list):
        bp["kpis"] = []

    return bp

def drop_placeholder_tables_and_relationships(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp) if isinstance(bp, dict) else {}
    tables = bp.get("suggested_tables") or []
    if not isinstance(tables, list):
        tables = []

    def is_placeholder(name: str) -> bool:
        n = sanitize_identifier(name or "")
        if not n:
            return True
        if "unnamed" in n:
            return True
        if n in {"entity", "entities", "table", "tables", "null", "none"}:
            return True
        return False

    kept_tables = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = t.get("table")
        if is_placeholder(tn):
            continue
        cols = t.get("columns") or []
        norm_cols = [sanitize_identifier(c) for c in (cols if isinstance(cols, list) else []) if str(c).strip()]
        norm_cols = [c for c in norm_cols if c and c not in ("unnamed", "none", "null")]
        if len(norm_cols) <= 1 and ("id" in norm_cols):
            continue
        kept_tables.append(t)

    bp["suggested_tables"] = kept_tables
    valid = {sanitize_identifier(t.get("table")) for t in kept_tables if isinstance(t, dict)}

    rels = bp.get("relationships") or []
    if not isinstance(rels, list):
        rels = []
    kept_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        a = sanitize_identifier(r.get("from"))
        b = sanitize_identifier(r.get("to"))
        if not a or not b:
            continue
        if is_placeholder(a) or is_placeholder(b):
            continue
        if a not in valid or b not in valid:
            continue
        fk = sanitize_identifier(r.get("fk") or "")
        if fk in ("unnamed", "relates_to", "none", "null"):
            r = dict(r)
            r["fk"] = ""
        kept_rels.append(r)
    bp["relationships"] = kept_rels
    return bp

def _try_parse_constraint_dict(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        obj = ast.literal_eval(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

FK_REGEX_GENERIC = re.compile(
    r"foreign\s+key\s*\(\s*([a-zA-Z0-9_]+)\s*\)\s*references\s*([a-zA-Z0-9_]+)\s*\(\s*([a-zA-Z0-9_]+)\s*\)",
    re.IGNORECASE,
)

def infer_relationships_from_constraints_and_columns(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp) if isinstance(bp, dict) else {}
    tables = bp.get("suggested_tables") or []
    if not isinstance(tables, list):
        return bp

    table_cols: Dict[str, List[str]] = {}
    table_cons: Dict[str, List[str]] = {}
    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = sanitize_identifier(t.get("table"))
        cols = t.get("columns") or []
        if not isinstance(cols, list):
            cols = []
        cols = [sanitize_identifier(c) for c in cols if str(c).strip()]
        table_cols[tn] = cols

        cons = t.get("constraints") or []
        if not isinstance(cons, list):
            cons = []
        table_cons[tn] = [str(x) for x in cons]

    valid_tables = set(table_cols.keys())
    rels: List[Dict[str, Any]] = []
    seen: set = set()

    for src_t, cons in table_cons.items():
        for con in cons:
            d = _try_parse_constraint_dict(con)
            if d and str(d.get("type", "")).lower() in ("foreign_key", "fk"):
                cols = d.get("columns") or []
                ref = str(d.get("references") or "")
                if isinstance(cols, list) and cols and "." in ref:
                    fk_col = sanitize_identifier(cols[0])
                    ref_table, ref_col = ref.split(".", 1)
                    ref_table = sanitize_identifier(ref_table)
                    ref_col = sanitize_identifier(ref_col)
                    if ref_table in valid_tables and fk_col:
                        key = (ref_table, src_t, fk_col)
                        if key not in seen:
                            seen.add(key)
                            rels.append({
                                "from": ref_table,
                                "to": src_t,
                                "cardinality": "one-to-many",
                                "fk": fk_col,
                                "notes": "inferred_from_fk_constraint_dict",
                                "ref_column": ref_col,
                            })
                continue

            m = FK_REGEX_GENERIC.search(con)
            if m:
                fk_col = sanitize_identifier(m.group(1))
                ref_table = sanitize_identifier(m.group(2))
                ref_col = sanitize_identifier(m.group(3))
                if ref_table in valid_tables:
                    key = (ref_table, src_t, fk_col)
                    if key not in seen:
                        seen.add(key)
                        rels.append({
                            "from": ref_table,
                            "to": src_t,
                            "cardinality": "one-to-many",
                            "fk": fk_col,
                            "notes": "inferred_from_fk_constraint_regex",
                            "ref_column": ref_col,
                        })

    def target_table_from_fk(col: str) -> Optional[str]:
        if not col.endswith("_id"):
            return None
        base = sanitize_identifier(col[:-3])
        if base in valid_tables:
            return base
        if base.endswith("s") and base[:-1] in valid_tables:
            return base[:-1]
        if (base + "s") in valid_tables:
            return base + "s"
        return None

    for src_t, cols in table_cols.items():
        for c in cols:
            if c == "parent_id":
                key = (src_t, src_t, "parent_id")
                if key not in seen:
                    seen.add(key)
                    rels.append({
                        "from": src_t,
                        "to": src_t,
                        "cardinality": "one-to-many",
                        "fk": "parent_id",
                        "notes": "self_reference",
                    })
                continue

            ref_t = target_table_from_fk(c)
            if not ref_t:
                continue
            key = (ref_t, src_t, c)
            if key in seen:
                continue
            seen.add(key)
            rels.append({
                "from": ref_t,
                "to": src_t,
                "cardinality": "one-to-many",
                "fk": c,
                "notes": "inferred_from_fk_column",
            })

    for jt, cols in table_cols.items():
        fk_cols = [c for c in cols if c.endswith("_id")]
        if len(fk_cols) != 2:
            continue
        a = target_table_from_fk(fk_cols[0])
        b = target_table_from_fk(fk_cols[1])
        if not a or not b or a == b:
            continue
        key = ("m2m",) + tuple(sorted([a, b])) + (jt,)
        if key in seen:
            continue
        seen.add(key)
        rels.append({
            "from": a,
            "to": b,
            "cardinality": "many-to-many",
            "fk": jt,
            "notes": f"junction_table:{jt}",
        })

    existing = bp.get("relationships") or []
    meaningful = []
    if isinstance(existing, list):
        for r in existing:
            if not isinstance(r, dict):
                continue
            a = sanitize_identifier(r.get("from"))
            b = sanitize_identifier(r.get("to"))
            if a in valid_tables and b in valid_tables:
                meaningful.append(r)
    bp["relationships"] = meaningful if meaningful else rels
    return bp

def infer_relationships_from_columns(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp) if isinstance(bp, dict) else {}
    tables = bp.get("suggested_tables") or []
    if not isinstance(tables, list) or not tables:
        return bp

    table_cols: Dict[str, List[str]] = {}
    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = sanitize_identifier(t.get("table"))
        cols = t.get("columns") or []
        if not isinstance(cols, list):
            cols = []
        cols = [sanitize_identifier(c) for c in cols if str(c).strip()]
        cols = [c for c in cols if c and c not in ("unnamed", "none", "null")]
        if tn:
            table_cols[tn] = cols

    valid = set(table_cols.keys())

    def resolve_table_name(base: str) -> Optional[str]:
        b = sanitize_identifier(base)
        if not b:
            return None
        if b in valid:
            return b
        if b.endswith("s") and b[:-1] in valid:
            return b[:-1]
        if (b + "s") in valid:
            return b + "s"
        if b.endswith("y") and (b[:-1] + "ies") in valid:
            return b[:-1] + "ies"
        if b.endswith("ies") and (b[:-3] + "y") in valid:
            return b[:-3] + "y"
        return None

    inferred = []
    seen = set()

    for child, cols in table_cols.items():
        for c in cols:
            if c == "parent_id":
                key = (child, child, "parent_id", "one-to-many")
                if key not in seen:
                    seen.add(key)
                    inferred.append({"from": child, "to": child, "cardinality": "one-to-many", "fk": "parent_id", "notes": "self_reference"})
                continue
            if not c.endswith("_id"):
                continue
            parent = resolve_table_name(c[:-3])
            if not parent:
                continue
            key = (parent, child, c, "one-to-many")
            if key in seen:
                continue
            seen.add(key)
            inferred.append({"from": parent, "to": child, "cardinality": "one-to-many", "fk": c, "notes": "inferred_from_fk_column"})

    for jt, cols in table_cols.items():
        fk_cols = [c for c in cols if c.endswith("_id")]
        if len(fk_cols) != 2:
            continue
        non_fk = [c for c in cols if c not in fk_cols]
        non_fk_nonmeta = [c for c in non_fk if c not in ("id", "created_at", "updated_at")]
        if len(non_fk_nonmeta) > 1:
            continue
        a = resolve_table_name(fk_cols[0][:-3])
        b = resolve_table_name(fk_cols[1][:-3])
        if not a or not b or a == b:
            continue
        key = tuple(sorted([a, b]) + [jt])
        if key in seen:
            continue
        seen.add(key)
        inferred.append({"from": a, "to": b, "cardinality": "many-to-many", "fk": jt, "notes": f"junction_table:{jt}"})

    existing = bp.get("relationships") or []
    if not isinstance(existing, list):
        existing = []
    meaningful = []
    for r in existing:
        if not isinstance(r, dict):
            continue
        a = sanitize_identifier(r.get("from"))
        b = sanitize_identifier(r.get("to"))
        if a in valid and b in valid:
            meaningful.append(r)

    if (len(meaningful) == 0) or (len(meaningful) <= 1 and len(inferred) >= 2):
        bp["relationships"] = inferred
    else:
        merged = meaningful[:]
        seen2 = {(sanitize_identifier(r.get("from")), sanitize_identifier(r.get("to")), sanitize_identifier(r.get("fk") or ""), canon_cardinality(r.get("cardinality","one-to-many"))) for r in meaningful if isinstance(r, dict)}
        for r in inferred:
            key = (r["from"], r["to"], sanitize_identifier(r.get("fk") or ""), canon_cardinality(r.get("cardinality","one-to-many")))
            if key not in seen2:
                merged.append(r)
                seen2.add(key)
        bp["relationships"] = merged

    return bp

def normalize_relationships(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp) if isinstance(bp, dict) else {}
    rels = bp.get("relationships") or []
    if not isinstance(rels, list):
        rels = []
    out = []
    seen = set()
    for r in rels:
        if not isinstance(r, dict):
            continue
        f = _safe_ident(r.get("from"))
        t = _safe_ident(r.get("to"))
        fk = _safe_ident(r.get("fk") or "")
        if not f or not t:
            continue
        card = canon_cardinality(r.get("cardinality", "one-to-many"))
        key = (f, t, fk, card)
        if key in seen:
            continue
        seen.add(key)
        out.append({"from": f, "to": t, "cardinality": card, "fk": fk, "notes": str(r.get("notes") or "")})
    bp["relationships"] = out
    return bp

def ensure_pk_every_table(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp)
    for t in bp.get("suggested_tables", []) or []:
        if not isinstance(t, dict):
            continue
        cons = [str(x) for x in (t.get("constraints") or [])]
        low = " ".join(c.lower() for c in cons)
        if "primary key" not in low:
            cols = t.get("columns") or []
            if "id" not in cols:
                t["columns"] = ["id"] + cols
            cons.append("primary key (id)")
            t["constraints"] = cons
    return bp

def add_missing_tables_from_relationships(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp)
    rels = bp.get("relationships") or []
    if not isinstance(rels, list) or not rels:
        return bp

    tables: Dict[str, Dict[str, Any]] = {
        t["table"]: t for t in (bp.get("suggested_tables", []) or [])
        if isinstance(t, dict) and t.get("table")
    }

    def ensure_table(name: str):
        name = _safe_ident(name)
        if name not in tables:
            tables[name] = {"table": name, "columns": ["id"], "constraints": ["primary key (id)"]}

    def ensure_col(tn: str, col: str):
        tn = _safe_ident(tn)
        col = _safe_ident(col)
        ensure_table(tn)
        if col and col not in tables[tn]["columns"]:
            tables[tn]["columns"].append(col)

    def add_fk(src: str, fk: str, ref: str, ref_col: str = "id"):
        src = _safe_ident(src)
        ref = _safe_ident(ref)
        fk = _safe_ident(fk)
        ref_col = _safe_ident(ref_col) or "id"
        ensure_col(src, fk)
        ensure_table(ref)
        con = f"foreign key ({fk}) references {ref} ({ref_col})"
        existing = " ".join([str(x).lower() for x in (tables[src].get("constraints") or [])])
        if con.lower() not in existing:
            tables[src].setdefault("constraints", []).append(con)

    for r in rels:
        if not isinstance(r, dict):
            continue
        a = _safe_ident(r.get("from"))
        b = _safe_ident(r.get("to"))
        card = canon_cardinality(r.get("cardinality", "one-to-many"))
        fk = _safe_ident(r.get("fk") or f"{a}_id")
        if not a or not b:
            continue

        ensure_table(a)
        ensure_table(b)

        if card == "many-to-many":
            j = _safe_ident("_".join(sorted([a, b])))
            ensure_table(j)
            add_fk(j, f"{a}_id", a, "id")
            add_fk(j, f"{b}_id", b, "id")
            jcons = tables[j].setdefault("constraints", [])
            pk = f"primary key ({a}_id, {b}_id)"
            if pk.lower() not in " ".join(c.lower() for c in jcons):
                jcons.append(pk)
        elif card == "many-to-one":
            add_fk(a, fk or f"{b}_id", b, "id")
        elif card == "one-to-many":
            add_fk(b, fk or f"{a}_id", a, "id")
        elif card == "one-to-one":
            add_fk(b, fk or f"{a}_id", a, "id")
            bcons = tables[b].setdefault("constraints", [])
            uq = f"unique ({_safe_ident(fk or f'{a}_id')})"
            if uq.lower() not in " ".join(c.lower() for c in bcons):
                bcons.append(uq)
        else:
            add_fk(b, fk or f"{a}_id", a, "id")

    bp["suggested_tables"] = list(tables.values())
    bp = ensure_pk_every_table(bp)
    return bp

def remove_conflicting_direct_vs_junction(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = dict(bp)
    tables = bp.get("suggested_tables", []) or []
    rels = bp.get("relationships", []) or []
    if not isinstance(tables, list) or not isinstance(rels, list):
        return bp

    tnames = {_safe_ident(t.get("table")) for t in tables if isinstance(t, dict)}
    junctions = {tn for tn in tnames if "_" in tn}
    has_m2m = any(canon_cardinality(r.get("cardinality")) == "many-to-many" for r in rels if isinstance(r, dict))

    if has_m2m:
        kept = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            a = _safe_ident(r.get("from"))
            b = _safe_ident(r.get("to"))
            card = canon_cardinality(r.get("cardinality", "one-to-many"))
            j1 = _safe_ident(f"{a}_{b}")
            j2 = _safe_ident(f"{b}_{a}")
            if (j1 in junctions or j2 in junctions) and card in ("one-to-many", "many-to-one", "one-to-one"):
                continue
            kept.append(r)
        bp["relationships"] = kept
        return bp

    new_tables = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = _safe_ident(t.get("table"))
        if tn in junctions:
            cols = set(_safe_ident(c) for c in (t.get("columns") or []))
            id_cols = [c for c in cols if c.endswith("_id")]
            if len(id_cols) == 2 and len(cols) <= 3:
                continue
        new_tables.append(t)
    bp["suggested_tables"] = new_tables
    return bp

def validate_blueprint(bp: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    bp = bp if isinstance(bp, dict) else {}
    tables = bp.get("suggested_tables") or []
    if not tables:
        errors.append("No suggested_tables produced.")
        return False, errors

    table_names = {_safe_ident(t.get("table")) for t in tables if isinstance(t, dict) and t.get("table")}
    if not table_names:
        errors.append("suggested_tables exist but no valid table names.")
        return False, errors

    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = _safe_ident(t.get("table"))
        cols = t.get("columns") or []
        if not cols:
            errors.append(f"Table '{tn}' has no columns.")
            continue
        cons = " ".join([str(x).lower() for x in (t.get("constraints") or [])])
        if "primary key" not in cons:
            errors.append(f"Table '{tn}' missing PRIMARY KEY constraint.")

    for t in tables:
        if not isinstance(t, dict):
            continue
        src = _safe_ident(t.get("table"))
        for con in t.get("constraints", []) or []:
            m = FK_RE.search(str(con))
            if m:
                ref_table = _safe_ident(m.group(2))
                if ref_table not in table_names:
                    errors.append(f"FK references unknown table '{ref_table}' from '{src}'.")

    tbls = bp.get('suggested_tables') or []
    names = [sanitize_identifier(t.get('table')) for t in tbls if isinstance(t, dict)]
    if any(('unnamed' in n) or (n in {'entity','entities','table','tables'}) for n in names):
        errors.append('Placeholder table detected (unnamed/entity).')
    rels = bp.get('relationships') or []
    if isinstance(rels, list) and len(rels) > 0:
        bad = 0
        good = 0
        name_set = set(names)
        for r in rels:
            if not isinstance(r, dict):
                continue
            a = sanitize_identifier(r.get('from'))
            b = sanitize_identifier(r.get('to'))
            if (not a) or (not b) or ('unnamed' in a) or ('unnamed' in b) or (a not in name_set) or (b not in name_set):
                bad += 1
            else:
                good += 1
        if good == 0 and bad > 0:
            errors.append('All relationships are invalid/placeholder. Use HITL review (Step 2/4).')

    return len(errors) == 0, errors

def synthesize_schema_from_er_llm(client: Optional[InferenceClient], er_brief: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a senior Data Architect. Convert the approved ER brief into a normalized relational schema.\n"
        "Return ONLY valid JSON with keys: suggested_tables, relationships, assumptions, kpis.\n\n"
        "Constraints:\n"
        "- suggested_tables MUST be non-empty.\n"
        "- Each table: {table, columns (>=2), constraints (include PRIMARY KEY)}.\n"
        "- Use junction tables for many-to-many.\n"
        "- Do NOT model both a junction and a direct FK for the same pair.\n"
        "- Use snake_case.\n"
        "- Translate business_rules into CHECK/UNIQUE/FK where reasonable; otherwise add to assumptions.\n"
        "No markdown."
    )
    raw, err = safe_chat_completion(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(er_brief, ensure_ascii=False)}],
        max_tokens=1900,
    )
    if err or raw is None:
        return {"_llm_error": err or "UNKNOWN", "suggested_tables": [], "relationships": [], "assumptions": [], "kpis": []}
    bp = parse_llm_json_or_repair(client, raw, max_repairs=2)
    return bp

NATURAL_KEY_HINTS = [
    "ean13", "ean", "ean_code", "identifikacni_cislo", "osobni_cislo",
    "kod", "kod_", "code", "cislo", "number", "uuid"
]

def pick_pk_for_entity(entity_name: str, attrs: List[str]) -> str:
    a = [sanitize_identifier(x) for x in (attrs or []) if str(x).strip()]
    for x in a:
        if x in ("id",):
            return "id"
    for hint in NATURAL_KEY_HINTS:
        for x in a:
            if x == hint or x.startswith(hint) or hint in x:
                return x
    return "id"

def fk_col_name(ref_entity: str, ref_pk: str) -> str:
    ref_entity = sanitize_identifier(ref_entity)
    ref_pk = sanitize_identifier(ref_pk) or "id"
    if ref_pk == "id":
        return f"{ref_entity}_id"
    return f"{ref_entity}_{ref_pk}"

def deterministic_schema_from_er(er: Dict[str, Any]) -> Dict[str, Any]:
    er = normalize_er_brief(er)
    ents = er.get("entities", [])
    rels = er.get("relationships", [])
    rules = er.get("business_rules", [])
    assumptions = er.get("assumptions", [])

    pk_map: Dict[str, str] = {}
    tables: Dict[str, Dict[str, Any]] = {}
    for e in ents:
        name = sanitize_identifier(e.get("name"))
        attrs = e.get("attributes") or []
        if not name:
            continue
        pk = pick_pk_for_entity(name, attrs)
        pk_map[name] = pk

        cols = [pk]
        for a in attrs:
            a = sanitize_identifier(a)
            if not a or a == pk:
                continue
            cols.append(a)
        if len(cols) == 1:
            cols.append("name")

        cons = [f"primary key ({pk})"]
        tables[name] = {"table": name, "columns": cols, "constraints": cons}

    def ensure_col(tn: str, col: str):
        t = tables.get(tn)
        if not t:
            pk = "id"
            pk_map[tn] = pk
            tables[tn] = {"table": tn, "columns": [pk, col], "constraints": [f"primary key ({pk})"]}
            return
        col = sanitize_identifier(col)
        if col and col not in t["columns"]:
            t["columns"].append(col)

    def add_fk(src_table: str, ref_table: str, unique: bool = False):
        src_table = sanitize_identifier(src_table)
        ref_table = sanitize_identifier(ref_table)
        ref_pk = pk_map.get(ref_table, "id")
        fk = fk_col_name(ref_table, ref_pk)
        ensure_col(src_table, fk)
        con = f"foreign key ({fk}) references {ref_table} ({ref_pk})"
        tables[src_table]["constraints"].append(con)
        if unique:
            tables[src_table]["constraints"].append(f"unique ({fk})")
        return fk

    out_rels: List[Dict[str, Any]] = []

    for r in rels:
        if not isinstance(r, dict):
            continue
        a = sanitize_identifier(r.get("from"))
        b = sanitize_identifier(r.get("to"))
        if not a or not b:
            continue
        card = canon_cardinality(r.get("cardinality", "one-to-many"))
        notes = str(r.get("notes") or "")

        if card == "one-to-many":
            fk = add_fk(b, a, unique=False)
            out_rels.append({"from": a, "to": b, "cardinality": "one-to-many", "fk": fk, "notes": notes})
        elif card == "many-to-one":
            fk = add_fk(a, b, unique=False)
            out_rels.append({"from": b, "to": a, "cardinality": "one-to-many", "fk": fk, "notes": notes})
        elif card == "one-to-one":
            fk = add_fk(b, a, unique=True)
            out_rels.append({"from": a, "to": b, "cardinality": "one-to-one", "fk": fk, "notes": notes})
        else:
            left, right = sorted([a, b])
            jt = f"{left}_{right}"
            lpk, rpk = pk_map.get(left, "id"), pk_map.get(right, "id")
            lfk = fk_col_name(left, lpk)
            rfk = fk_col_name(right, rpk)

            if jt not in tables:
                tables[jt] = {
                    "table": jt,
                    "columns": [lfk, rfk],
                    "constraints": [
                        f"primary key ({lfk}, {rfk})",
                        f"foreign key ({lfk}) references {left} ({lpk})",
                        f"foreign key ({rfk}) references {right} ({rpk})",
                    ],
                }
            out_rels.append({"from": left, "to": jt, "cardinality": "one-to-many", "fk": lfk, "notes": "junction"})
            out_rels.append({"from": right, "to": jt, "cardinality": "one-to-many", "fk": rfk, "notes": "junction"})

    bp = {
        "suggested_tables": list(tables.values()),
        "relationships": out_rels,
        "assumptions": [str(x).strip() for x in (assumptions or []) if str(x).strip()],
        "kpis": [],
    }
    for x in (rules or []):
        x = str(x).strip()
        if x:
            bp["assumptions"].append(f"Business rule to enforce: {x}")
    return bp

def repair_schema_llm(client: Optional[InferenceClient], bp: Dict[str, Any], errors: List[str], er_brief: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a senior Data Architect fixing a relational schema.\n"
        "Return ONLY valid JSON with keys: suggested_tables, relationships, assumptions, kpis.\n"
        "Fix validation errors and keep the schema implementable.\n"
        "Do NOT return empty suggested_tables.\n"
        "No markdown."
    )
    payload = {"errors": errors, "current_schema": bp, "er_brief": er_brief}
    raw, err = safe_chat_completion(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        max_tokens=1900,
    )
    if err or raw is None:
        bp2 = dict(bp)
        bp2["_llm_error"] = err or "UNKNOWN"
        return bp2
    fixed = parse_llm_json_or_repair(client, raw, max_repairs=2)
    return fixed

def postprocess_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    bp = drop_invalid_tables(bp)
    bp = drop_placeholder_tables_and_relationships(bp)
    bp = normalize_relationships(bp)
    bp = add_missing_tables_from_relationships(bp)
    bp = remove_conflicting_direct_vs_junction(bp)
    bp = ensure_pk_every_table(bp)
    for t in bp.get('suggested_tables', []) or []:
        if not isinstance(t, dict):
            continue
        cons = [str(x).strip() for x in (t.get('constraints') or []) if str(x).strip()]
        seen = set()
        out = []
        for c in cons:
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        t['constraints'] = out
    bp = normalize_relationships(bp)
    return bp

def infer_relationships_from_fk_constraints(bp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rels = []
    for t in bp.get("suggested_tables", []) or []:
        src = _safe_ident(t.get("table"))
        for con in t.get("constraints", []) or []:
            m = FK_RE.search(str(con))
            if not m:
                continue
            src_col, ref_table, ref_col = m.group(1), m.group(2), m.group(3)
            rels.append({"from": _safe_ident(ref_table), "to": src, "cardinality": "one-to-many", "fk": _safe_ident(src_col), "notes": "inferred"})
    return rels

def build_mermaid_from_blueprint(bp: Dict[str, Any]) -> str:
    tables = bp.get("suggested_tables", []) or []
    rels = bp.get("relationships", []) or []
    inferred = infer_relationships_from_fk_constraints(bp)
    all_rels = (rels if isinstance(rels, list) else []) + inferred

    m = "erDiagram\n"
    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = _title_ident(t.get("table"))
        cols = t.get("columns", []) or []
        if not tn or not cols:
            continue
        m += f"    {tn} {{\n"
        pk_cols = set()
        for con in t.get("constraints", []) or []:
            s = str(con).lower()
            if "primary key" in s:
                inside = re.search(r"primary\s+key\s*\(([^\)]+)\)", s)
                if inside:
                    pk_cols |= {sanitize_identifier(x) for x in inside.group(1).split(",")}
        for col in cols:
            c = sanitize_identifier(col)
            pk = " PK" if c in pk_cols else ""
            m += f"        STRING {c}{pk}\n"
        m += "    }\n"

    for r in all_rels:
        if not isinstance(r, dict):
            continue
        a = _title_ident(r.get("from"))
        b = _title_ident(r.get("to"))
        if not a or not b:
            continue
        card = canon_cardinality(r.get("cardinality", "one-to-many"))
        if card == "one-to-one":
            op = "||--||"
        elif card == "many-to-many":
            op = "}o--o{"
        else:
            op = "||--o{"
        fk = sanitize_identifier(r.get("fk") or "")
        if not fk or fk in ("unnamed", "relates_to"):
            fk = f"{sanitize_identifier(r.get('from') or '')}_id"
            fk = fk if fk and fk != "_id" else "id"
        lbl = fk
        m += f'    {a} {op} {b} : "{lbl}"\n'
    return m

def build_sql_from_blueprint(bp: Dict[str, Any]) -> str:
    tables = bp.get("suggested_tables", []) or []
    ddl = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        tn = _safe_ident(t.get("table"))
        cols = t.get("columns", []) or []
        cons = t.get("constraints", []) or []
        if not tn or not cols:
            continue

        col_lines = []
        for col in cols:
            c = _safe_ident(col)
            if c == "id" or c.endswith("_id"):
                typ = "VARCHAR(64)"
            elif any(k in c for k in ["date", "datum", "created_at", "updated_at", "valid_from", "valid_to"]):
                typ = "TIMESTAMP"
            elif any(k in c for k in ["qty", "quantity", "amount", "price", "total", "score", "age"]):
                typ = "NUMERIC"
            else:
                typ = "TEXT"
            col_lines.append(f'  "{c}" {typ}')

        other_lines = []
        for con in cons:
            s = str(con).strip()
            low = s.lower()
            if low.startswith(("primary key", "foreign key", "unique", "check")):
                other_lines.append("  " + s)

        ddl.append(f'CREATE TABLE "{tn}" (\n' + ",\n".join(col_lines + other_lines) + "\n);\n")

    return "\n".join(ddl)

# ============================================================
# HITL conversions for wizard editors
# ============================================================

def er_entities_to_df(er: Dict[str, Any]) -> pd.DataFrame:
    ents = er.get("entities", []) if isinstance(er, dict) else []
    rows = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        rows.append({"name": e.get("name",""), "description": e.get("description",""), "attributes": ", ".join(e.get("attributes", []) or [])})
    if not rows:
        return pd.DataFrame(columns=["name","description","attributes"])
    return pd.DataFrame(rows)

def er_relationships_to_df(er: Dict[str, Any]) -> pd.DataFrame:
    rels = er.get("relationships", []) if isinstance(er, dict) else []
    rows = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        rows.append({
            "from": r.get("from",""),
            "to": r.get("to",""),
            "cardinality": canon_cardinality(r.get("cardinality","one-to-many")),
            "fk_on": r.get("fk_on",""),
            "fk_name": r.get("fk_name",""),
            "notes": r.get("notes",""),
        })
    if not rows:
        return pd.DataFrame(columns=["from","to","cardinality","fk_on","fk_name","notes"])
    return pd.DataFrame(rows)

def df_to_er(ent_df: pd.DataFrame, rel_df: pd.DataFrame, rules_text: str, assumptions_text: str) -> Dict[str, Any]:
    ents = []
    if isinstance(ent_df, pd.DataFrame) and not ent_df.empty:
        for _, row in ent_df.iterrows():
            name = sanitize_identifier(row.get("name",""))
            if not name:
                continue
            attrs = [sanitize_identifier(x) for x in str(row.get("attributes","")).split(",") if x.strip()]
            ents.append({"name": name, "description": str(row.get("description",""))[:400], "attributes": attrs})

    rels = []
    if isinstance(rel_df, pd.DataFrame) and not rel_df.empty:
        for _, row in rel_df.iterrows():
            f = sanitize_identifier(row.get("from",""))
            t = sanitize_identifier(row.get("to",""))
            if not f or not t:
                continue
            rels.append({
                "from": f,
                "to": t,
                "cardinality": canon_cardinality(row.get("cardinality","one-to-many")),
                "fk_on": sanitize_identifier(row.get("fk_on","")),
                "fk_name": sanitize_identifier(row.get("fk_name","")),
                "notes": str(row.get("notes",""))[:500],
            })

    rules = [x.strip() for x in (rules_text or "").splitlines() if x.strip()]
    assm = [x.strip() for x in (assumptions_text or "").splitlines() if x.strip()]

    return normalize_er_brief({"entities": ents, "relationships": rels, "business_rules": rules, "assumptions": assm})

def blueprint_to_editable_tables(bp: Dict[str, Any]) -> pd.DataFrame:
    tables = bp.get("suggested_tables", []) if isinstance(bp, dict) else []
    rows = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        rows.append({"table": t.get("table",""), "columns": ", ".join(t.get("columns", []) or []), "constraints": "\n".join(t.get("constraints", []) or [])})
    if not rows:
        return pd.DataFrame(columns=["table", "columns", "constraints"])
    return pd.DataFrame(rows)

def editable_tables_to_blueprint_tables(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    for _, row in df.iterrows():
        tn = sanitize_identifier(row.get("table",""))
        if not tn:
            continue
        cols = [sanitize_identifier(x) for x in str(row.get("columns","")).split(",") if x.strip()]
        cons = [x.strip() for x in str(row.get("constraints","")).splitlines() if x.strip()]
        if cols:
            out.append({"table": tn, "columns": cols, "constraints": cons})
    return out

def blueprint_to_editable_relationships(bp: Dict[str, Any]) -> pd.DataFrame:
    rels = bp.get("relationships", []) if isinstance(bp, dict) else []
    rows = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        rows.append({"from": r.get("from",""), "to": r.get("to",""), "cardinality": canon_cardinality(r.get("cardinality","one-to-many")), "fk": r.get("fk",""), "notes": r.get("notes","")})
    if not rows:
        return pd.DataFrame(columns=["from","to","cardinality","fk","notes"])
    return pd.DataFrame(rows)

def editable_relationships_to_blueprint(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    for _, row in df.iterrows():
        f = sanitize_identifier(row.get("from",""))
        t = sanitize_identifier(row.get("to",""))
        if not f or not t:
            continue
        out.append({"from": f, "to": t, "cardinality": canon_cardinality(row.get("cardinality","one-to-many")), "fk": sanitize_identifier(row.get("fk","")), "notes": str(row.get("notes",""))[:800]})
    return out

# ============================================================
# CSV modelling (HITL)
# ============================================================

def infer_sql_type(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "TEXT"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"

    sample = s.astype(str).head(200)
    dt = pd.to_datetime(sample, errors="coerce", utc=False)
    if dt.notna().mean() > 0.9:
        return "TIMESTAMP"

    num = pd.to_numeric(sample, errors="coerce")
    if num.notna().mean() > 0.9:
        if np.all(np.isclose(num.dropna() % 1, 0)):
            return "BIGINT"
        return "DOUBLE PRECISION"

    max_len = int(sample.map(len).max())
    if max_len <= 255:
        return f"VARCHAR({max(1, max_len)})"
    return "TEXT"

def profile_table(df: pd.DataFrame, sample_n: int = 1000) -> Dict[str, Any]:
    n = len(df)
    out = {"rows": int(n), "columns": []}
    for c in df.columns:
        ser = df[c]
        non_null_ratio = float(ser.notna().mean()) if n else 0.0
        nn = ser.dropna()
        uniq_ratio = float(nn.nunique() / max(1, len(nn))) if len(nn) else 0.0
        sample = nn
        if len(sample) > sample_n:
            sample = sample.sample(sample_n, random_state=42)
        samples = sample.astype(str).head(8).tolist()
        out["columns"].append({
            "column": c,
            "suggested_type": infer_sql_type(ser),
            "nullable": bool(ser.isna().any()),
            "non_null_ratio": non_null_ratio,
            "unique_ratio": uniq_ratio,
            "samples": samples,
        })
    return out

def pk_candidates(df: pd.DataFrame, max_k: int = 3) -> List[str]:
    n = len(df)
    if n == 0:
        return []
    cands = []
    for c in df.columns:
        ser = df[c]
        if ser.isna().any():
            continue
        if ser.nunique(dropna=False) == n:
            cands.append(c)
    return cands[:max_k]

def _canon_values(series: pd.Series, limit: int = 20000) -> List[str]:
    s = series.dropna()
    if s.empty:
        return []
    if len(s) > limit:
        s = s.sample(limit, random_state=42)
    out = []
    for v in s.astype(str).tolist():
        v = v.strip()
        if re.fullmatch(r"-?\d+\.0+", v):
            v = v.split(".")[0]
        out.append(v)
    return out

def containment_ratio(src: pd.Series, ref: pd.Series, limit: int = 20000) -> float:
    src_vals = _canon_values(src, limit)
    ref_vals = set(_canon_values(ref, limit))
    if not src_vals or not ref_vals:
        return 0.0
    return float(np.mean([v in ref_vals for v in src_vals]))

def infer_relationships_csv(
    tables: Dict[str, pd.DataFrame],
    schema: Dict[str, List[Dict[str, Any]]],
    threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    pk_map: Dict[str, List[str]] = {}
    for t, cols in schema.items():
        pk_cols = [c["column"] for c in cols if c.get("pk")]
        if pk_cols:
            pk_map[t] = pk_cols
    for t, df in tables.items():
        if t not in pk_map or not pk_map[t]:
            pk_map[t] = pk_candidates(df)

    rels = []
    for src_t, src_df in tables.items():
        for src_c in src_df.columns:
            if not (src_c.endswith("_id") or src_c == "id" or "id" in src_c):
                continue
            for ref_t, ref_df in tables.items():
                if ref_t == src_t:
                    continue
                for ref_pk in pk_map.get(ref_t, []):
                    name_match = (src_c == ref_pk) or (src_c == f"{ref_t}_id")
                    if not name_match:
                        continue
                    ratio = containment_ratio(src_df[src_c], ref_df[ref_pk])
                    if ratio >= threshold:
                        fk_ser = src_df[src_c].dropna()
                        is_unique_fk = (fk_ser.nunique() == len(fk_ser)) if len(fk_ser) else False
                        rels.append({
                            "approved": False,
                            "ref_table": ref_t,
                            "ref_column": ref_pk,
                            "src_table": src_t,
                            "src_column": src_c,
                            "cardinality": "1:1" if is_unique_fk else "1:N",
                            "containment": float(ratio),
                            "label": "relates_to",
                        })
    rels.sort(key=lambda r: r["containment"], reverse=True)
    return rels

def build_sql_from_schema(schema_frames: Dict[str, pd.DataFrame], rels: List[Dict[str, Any]]) -> str:
    sql = ""
    for t, df_cols in schema_frames.items():
        cols_sql = []
        pks = df_cols[df_cols["pk"] == True]["column"].tolist()
        for _, r in df_cols.iterrows():
            null_sql = "" if bool(r["nullable"]) else " NOT NULL"
            cols_sql.append(f'  "{r["column"]}" {r["type"]}{null_sql}')
        if pks:
            pk_str = ", ".join([f'"{p}"' for p in pks])
            cols_sql.append(f"  PRIMARY KEY ({pk_str})")
        sql += f'CREATE TABLE "{t}" (\n' + ",\n".join(cols_sql) + "\n);\n\n"
    for r in rels:
        if not r.get("approved"):
            continue
        sql += (
            f'ALTER TABLE "{r["src_table"]}" '
            f'ADD CONSTRAINT "fk_{r["src_table"]}_{r["src_column"]}" '
            f'FOREIGN KEY ("{r["src_column"]}") '
            f'REFERENCES "{r["ref_table"]}"("{r["ref_column"]}");\n'
        )
    return sql

def build_mermaid_from_schema(schema_frames: Dict[str, pd.DataFrame], rels: List[Dict[str, Any]]) -> str:
    m = "erDiagram\n"
    for t, df_cols in schema_frames.items():
        m += f"    {_title_ident(t)} {{\n"
        for _, r in df_cols.iterrows():
            clean_type = str(r["type"]).split("(")[0].replace(" ", "_").strip()
            pk = " PK" if bool(r["pk"]) else ""
            m += f"        {clean_type} {sanitize_identifier(r['column'])}{pk}\n"
        m += "    }\n"
    for r in rels:
        if not r.get("approved"):
            continue
        op = "||--o{" if r["cardinality"] == "1:N" else "||--||"
        lbl = str(r.get("label") or "relates_to").replace('"', "")
        m += f"    {_title_ident(r['ref_table'])} {op} {_title_ident(r['src_table'])} : \"{lbl}\"\n"
    return m

# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="AI Data Modeling Agent (Wizard + HITL)", layout="wide")
st.title("🏗️ AI Data Modeling Agent — Wizard + HITL")
audit_init()

hf_token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

if "llm_available" not in st.session_state:
    st.session_state["llm_available"] = True

client: Optional[InferenceClient] = None
if hf_token:
    client = InferenceClient(provider="auto", model=MODEL_ID, api_key=hf_token)
else:
    st.session_state["llm_available"] = False

with st.sidebar:
    st.markdown("### 🧠 LLM")
    st.code(f"Model: {MODEL_ID}")
    if hf_token and st.session_state.get("llm_available", True):
        st.success("✅ LLM ready")
    elif hf_token and not st.session_state.get("llm_available", True):
        st.warning("⚠️ LLM calls failing (402/HTTP). HITL + deterministic fixes still work.")
    else:
        st.error("❌ Missing HUGGINGFACE_TOKEN")

    st.markdown("---")
    st.markdown("### 📄 Report & Exports")
    if st.button("📊 Generate PDF Report"):
        audit_snapshot_data = audit_snapshot()
        pdf_bytes = build_pdf_report_bytes(audit_snapshot_data)
        audit_event("global", "report_generated", {"bytes": len(pdf_bytes)})
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"ai_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

tab_business, tab_csv = st.tabs(["🧩 Business Wizard (ER → Schema)", "📁 CSV Modeling (HITL)"])

with tab_business:
    st.subheader("Business Wizard — step by step")

    st.session_state.setdefault("biz_step", 1)
    st.session_state.setdefault("biz_er_approved", False)
    st.session_state.setdefault("biz_er_brief", None)
    st.session_state.setdefault("biz_schema", None)

    with st.sidebar:
        st.markdown("---")
        st.markdown("## Business Wizard")
        step = st.radio(
            "Step",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1) Extract ER (LLM)",
                2: "2) Review & Approve (HITL)",
                3: "3) Generate Schema (LLM)",
                4: "4) Final Review (HITL)",
                5: "5) Export",
            }[x],
            index=st.session_state["biz_step"] - 1,
            key="biz_step_radio",
        )
        st.session_state["biz_step"] = step

        st.write("✅ ER approved" if st.session_state.get("biz_er_approved") else "⬜ ER not approved")
        st.write("✅ Schema generated" if st.session_state.get("biz_schema") else "⬜ Schema not generated")

        if st.button("🔄 Reset Business Wizard"):
            for k in [
                "biz_step", "biz_er_approved", "biz_er_brief", "biz_schema",
                "biz_task_text",
                "biz_ent_df", "biz_rel_df", "biz_rules_text", "biz_assumptions_text",
                "biz_tables_df", "biz_bp_rels_df",
                "biz_validation",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            audit_event("business", "reset_wizard", {})
            st.rerun()

    if st.session_state["biz_step"] >= 3 and not st.session_state.get("biz_er_approved"):
        st.warning("⚠️ Please approve entities and relations in Step 2 first")
        st.session_state["biz_step"] = 2
        st.rerun()

    if st.session_state["biz_step"] == 5 and not st.session_state.get("biz_schema"):
        st.warning("⚠️ Please generate the schema in Step 3 first")
        st.session_state["biz_step"] = 3
        st.rerun()

    current_step = st.session_state["biz_step"]

    if current_step == 1:
        st.markdown("### Step 1 — LLM extracts entities and relations (ER brief)")
        business_task = st.text_area(
            "Assignment:",
            height=220,
            key="biz_task_text",
            placeholder="Describe the business problem/domain. The more rules, the better the design.",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🧠 Extract ER (LLM)", use_container_width=True):
                if not business_task.strip():
                    st.warning("Enter your assignment.")
                else:
                    with st.spinner("Extracting ER..."):
                        audit_set_task(business_task)
                        er = extract_er_brief_llm(client if st.session_state.get("llm_available", True) else None, business_task)
                        audit_event("business", "extract_er", er)
                        st.session_state["biz_er_brief"] = er
                        st.session_state["biz_er_approved"] = False
                        st.session_state["biz_ent_df"] = er_entities_to_df(er)
                        st.session_state["biz_rel_df"] = er_relationships_to_df(er)
                        st.session_state["biz_rules_text"] = "\n".join(er.get("business_rules", []) or [])
                        st.session_state["biz_assumptions_text"] = "\n".join(er.get("assumptions", []) or [])
                        st.session_state["biz_step"] = 2
                        st.rerun()
        with c2:
            st.info("💡 Tip: If LLM doesn't know, it leaves it in assumptions. You can correct this in step 2.")

        er_preview = st.session_state.get("biz_er_brief")
        if er_preview:
            with st.expander("Preview ER brief (JSON)"):
                st.json(er_preview)

    elif current_step == 2:
        st.markdown("### Step 2 — Edit and approve ER (HITL)")
        er = st.session_state.get("biz_er_brief") or {"entities": [], "relationships": [], "business_rules": [], "assumptions": []}
        st.caption("You only deal with the concept here: entities + relationships + rules. Tables are generated only after approval.")

        ent_df = st.session_state.get("biz_ent_df", er_entities_to_df(er))
        rel_df = st.session_state.get("biz_rel_df", er_relationships_to_df(er))
        rules_text = st.session_state.get("biz_rules_text", "\n".join(er.get("business_rules", []) or []))
        assumptions_text = st.session_state.get("biz_assumptions_text", "\n".join(er.get("assumptions", []) or []))

        left, right = st.columns(2)
        with left:
            st.markdown("**Entities**")
            ent_df = st.data_editor(ent_df, use_container_width=True, num_rows="dynamic", key="biz_ent_editor")
            if not ent_df.empty:
                ent_df["name"] = ent_df["name"].astype(str).map(sanitize_identifier)
        with right:
            st.markdown("**Relationships**")
            rel_df = st.data_editor(rel_df, use_container_width=True, num_rows="dynamic", key="biz_rel_editor")
            if not rel_df.empty:
                rel_df["from"] = rel_df["from"].astype(str).map(sanitize_identifier)
                rel_df["to"] = rel_df["to"].astype(str).map(sanitize_identifier)
                rel_df["fk_on"] = rel_df["fk_on"].astype(str).map(sanitize_identifier)
                rel_df["fk_name"] = rel_df["fk_name"].astype(str).map(sanitize_identifier)
                rel_df["cardinality"] = rel_df["cardinality"].astype(str).map(canon_cardinality)

        st.markdown("**Business rules**")
        rules_text = st.text_area("1 line = 1 rule", value=rules_text, height=120, key="biz_rules_area")

        st.markdown("**Assumptions (what LLM estimated)**")
        assumptions_text = st.text_area("1 row = 1 assumption", value=assumptions_text, height=100, key="biz_assumptions_area")

        st.session_state["biz_ent_df"] = ent_df
        st.session_state["biz_rel_df"] = rel_df
        st.session_state["biz_rules_text"] = rules_text
        st.session_state["biz_assumptions_text"] = assumptions_text

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("💾 Save ER draft", use_container_width=True):
                er2 = df_to_er(ent_df, rel_df, rules_text, assumptions_text)
                st.session_state["biz_er_brief"] = er2
                st.session_state["biz_er_approved"] = False
                audit_event("business", "er_saved_draft", er2)
                st.success("Saved (not yet approved).")

        with c2:
            if st.button("✅ Approve ER → Continue", use_container_width=True):
                er2 = df_to_er(ent_df, rel_df, rules_text, assumptions_text)
                st.session_state["biz_er_brief"] = er2
                st.session_state["biz_er_approved"] = True
                audit_event("business", "er_approved", er2)
                st.session_state["biz_step"] = 3
                st.rerun()

        with c3:
            st.info("ℹ️ Approval locks the direction: only then is the schema (tables, PK/FK) generated.")

        with st.expander("ER brief JSON"):
            st.json(st.session_state.get("biz_er_brief"))

    elif current_step == 3:
        st.markdown("### Step 3 — Generate schema (tables, PK/FK)")
        er = st.session_state.get("biz_er_brief")
        if not er:
            st.error("ER brief is missing. Return to Step 1.")
        else:
            st.caption("After generation, deterministic correction (PK/FK/junction conflicts) and validation are performed...")

            use_llm_schema = st.checkbox('Use LLM also for schema design (less stable)', value=False)

            if st.button("🏗️ Generate Schema", use_container_width=True):
                with st.spinner("Generating schema..."):
                    bp = (synthesize_schema_from_er_llm(client if st.session_state.get("llm_available", True) else None, er)
                          if use_llm_schema else deterministic_schema_from_er(er))
                    audit_event("business", "schema_raw", bp)
                    bp = postprocess_blueprint(bp)
                    ok, errs = validate_blueprint(bp)
                    st.session_state["biz_schema"] = bp
                    st.session_state["biz_validation"] = {"ok": ok, "errors": errs}
                    st.session_state["biz_tables_df"] = blueprint_to_editable_tables(bp)
                    st.session_state["biz_bp_rels_df"] = blueprint_to_editable_relationships(bp)
                    audit_event("business", "schema_postprocess_validate", {"ok": ok, "errors": errs, **bp})
                    st.session_state["biz_step"] = 4
                    st.rerun()

            bp = st.session_state.get("biz_schema")
            if bp:
                val = st.session_state.get("biz_validation", {"ok": True, "errors": []})
                if val.get("ok"):
                    st.success("✅ Validation OK.")
                else:
                    st.error("❌ Validation failed:\n- " + "\n- ".join(val.get("errors", [])))
                with st.expander("Schema JSON"):
                    st.json(bp)

    elif current_step == 4:
        st.markdown("### Step 4 — Final review (HITL)")
        bp = st.session_state.get("biz_schema")
        er = st.session_state.get("biz_er_brief") or {}
        if not bp:
            st.error("The schema has not been generated yet. Go to Step 3.")
        else:
            val = st.session_state.get("biz_validation", {"ok": True, "errors": []})
            if not val.get("ok"):
                st.error("❌ Validation issues:\n- " + "\n- ".join(val.get("errors", [])))

            tables_df = st.session_state.get("biz_tables_df", blueprint_to_editable_tables(bp))
            rels_df = st.session_state.get("biz_bp_rels_df", blueprint_to_editable_relationships(bp))

            st.caption("Tables: write columns as `a, b, c`. Constraints: 1 row = 1 constraint.")
            tables_df = st.data_editor(tables_df, use_container_width=True, num_rows="dynamic", key="biz_tables_editor")

            st.caption("Relationships (ERD layer). If you want, leave them blank — ERD will be calculated from FK constraints.")
            rels_df = st.data_editor(rels_df, use_container_width=True, num_rows="dynamic", key="biz_schema_rels_editor")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                if st.button("💾 Save edits", use_container_width=True):
                    bp2 = dict(bp)
                    bp2["suggested_tables"] = editable_tables_to_blueprint_tables(tables_df)
                    bp2["relationships"] = editable_relationships_to_blueprint(rels_df)
                    bp2 = postprocess_blueprint(bp2)
                    ok, errs = validate_blueprint(bp2)
                    st.session_state["biz_schema"] = bp2
                    st.session_state["biz_validation"] = {"ok": ok, "errors": errs}
                    st.session_state["biz_tables_df"] = blueprint_to_editable_tables(bp2)
                    st.session_state["biz_bp_rels_df"] = blueprint_to_editable_relationships(bp2)
                    audit_event("business", "schema_hitl_saved", bp2)
                    st.rerun()

            with c2:
                if st.button("✅ Validate", use_container_width=True):
                    ok, errs = validate_blueprint(st.session_state.get("biz_schema", {}))
                    st.session_state["biz_validation"] = {"ok": ok, "errors": errs}
                    audit_event("business", "validate_manual", {"ok": ok, "errors": errs})
                    if ok:
                        st.success("✅ Validation passed!")
                    else:
                        st.error("❌ Did not pass:\n- " + "\n- ".join(errs))

            with c3:
                if st.button("🧹 Auto-fix", use_container_width=True):
                    bp2 = postprocess_blueprint(st.session_state.get("biz_schema", {}))
                    ok, errs = validate_blueprint(bp2)
                    st.session_state["biz_schema"] = bp2
                    st.session_state["biz_validation"] = {"ok": ok, "errors": errs}
                    st.session_state["biz_tables_df"] = blueprint_to_editable_tables(bp2)
                    st.session_state["biz_bp_rels_df"] = blueprint_to_editable_relationships(bp2)
                    audit_event("business", "auto_fix_deterministic", bp2)
                    st.rerun()

            with c4:
                if st.button("🛠️ Repair (LLM)", use_container_width=True):
                    if not st.session_state.get("llm_available", True) or client is None:
                        st.error("LLM not available.")
                    else:
                        bp2 = st.session_state.get("biz_schema", {})
                        ok, errs = validate_blueprint(bp2)
                        if ok:
                            st.info("✅ The schema is already valid.")
                        else:
                            with st.spinner("Repairing via LLM..."):
                                fixed = repair_schema_llm(client, bp2, errs, er)
                                audit_event("business", "repair_llm_raw", fixed)
                                fixed = postprocess_blueprint(fixed)
                                ok2, errs2 = validate_blueprint(fixed)
                                st.session_state["biz_schema"] = fixed
                                st.session_state["biz_validation"] = {"ok": ok2, "errors": errs2}
                                st.session_state["biz_tables_df"] = blueprint_to_editable_tables(fixed)
                                st.session_state["biz_bp_rels_df"] = blueprint_to_editable_relationships(fixed)
                                audit_event("business", "repair_llm_done", fixed)
                                st.rerun()

            mermaid = build_mermaid_from_blueprint(st.session_state.get("biz_schema", {}))
            ddl = build_sql_from_blueprint(st.session_state.get("biz_schema", {}))

            st.markdown("---")
            st.subheader("📐 Preview ERD")
            st.code(mermaid, language="text")
            render_mermaid(mermaid, height=650)
            audit_event("business", "schema_diagram", {"mermaid": mermaid})

            st.markdown("---")
            st.subheader("🗄️ Preview SQL DDL")
            st.code(ddl, language="sql")
            audit_event("business", "schema_ddl", {"ddl": ddl})

            st.markdown("---")
            if st.button("➡️ Continue to Export", use_container_width=True):
                st.session_state["biz_step"] = 5
                st.rerun()

    elif current_step == 5:
        st.markdown("### Step 5 — Export")
        bp = st.session_state.get("biz_schema") or {}
        if not bp:
            st.error("Missing schema. Go back to Step 3.")
        else:
            mermaid = build_mermaid_from_blueprint(bp)
            ddl = build_sql_from_blueprint(bp)
            st.success("✅ Done! Download the assets below.")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("📊 Download ERD (.mmd)", mermaid, file_name="business_erd.mmd", use_container_width=True)
            with c2:
                st.download_button("🗄️ Download DDL (.sql)", ddl, file_name="business_schema.sql", use_container_width=True)
            with c3:
                st.download_button("📋 Download JSON (.json)", json.dumps(bp, ensure_ascii=False, indent=2), file_name="business_blueprint.json", use_container_width=True)

            st.markdown("---")
            with st.expander("📐 ERD Mermaid"):
                st.code(mermaid, language="text")
                render_mermaid(mermaid, height=650)

            with st.expander("🗄️ SQL DDL"):
                st.code(ddl, language="sql")

            with st.expander("📋 Blueprint JSON"):
                st.json(bp)

with tab_csv:
    st.subheader("CSV Modeling (HITL)")
    uploaded_files = st.file_uploader("Upload CSV Sources", type="csv", accept_multiple_files=True)

    st.session_state.setdefault("tables", {})
    st.session_state.setdefault("profiles", {})
    st.session_state.setdefault("schema", {})
    st.session_state.setdefault("rel_candidates", [])
    st.session_state.setdefault("csv_iterations", [])

    if uploaded_files:
        for f in uploaded_files:
            t_name = sanitize_identifier(f.name.rsplit(".", 1)[0])
            df = pd.read_csv(f)
            df.columns = [sanitize_identifier(c) for c in df.columns]
            st.session_state.tables[t_name] = df

        audit_event("csv", "data_ingestion", {"tables": list(st.session_state.tables.keys())})

        for t_name, df in st.session_state.tables.items():
            if t_name not in st.session_state.profiles:
                st.session_state.profiles[t_name] = profile_table(df)
            if t_name not in st.session_state.schema:
                pks = pk_candidates(df)
                st.session_state.schema[t_name] = [
                    {"column": c["column"], "type": c["suggested_type"], "nullable": bool(c["nullable"]), "pk": (c["column"] in pks and len(pks) == 1)}
                    for c in st.session_state.profiles[t_name]["columns"]
                ]

        st.session_state.rel_candidates = infer_relationships_csv(st.session_state.tables, st.session_state.schema, threshold=0.95)
        audit_event("csv", "infer_relationships", st.session_state.rel_candidates[:50])

        with st.expander("🔍 Data preview", expanded=False):
            for t, df in st.session_state.tables.items():
                st.write(f"Table: `{t}`")
                st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Step 1: LLM-assisted Schema Patch (scoped)")
        all_tables = sorted(list(st.session_state.schema.keys()))
        target_tables = st.multiselect("Select tables for refinement:", options=all_tables, default=all_tables[: min(3, len(all_tables))])
        instruction = st.text_area("Instruction:", height=90)

        if st.button("🔄 Execute LLM Patch (CSV)", use_container_width=True):
            if not target_tables:
                st.warning("Select at least one table.")
            else:
                if not (st.session_state.get("llm_available", True) and client is not None):
                    st.error("LLM not available (token / HF issue).")
                else:
                    with st.spinner("Generating schema patch..."):
                        schema_subset = {t: [{"column": c["column"], "type": c["type"], "nullable": c["nullable"], "pk": c["pk"]} for c in st.session_state.schema[t]] for t in target_tables}
                        prof_subset = {
                            t: {
                                "rows": st.session_state.profiles[t]["rows"],
                                "columns": [{
                                    "column": c["column"],
                                    "suggested_type": c["suggested_type"],
                                    "nullable": c["nullable"],
                                    "unique_ratio": round(float(c["unique_ratio"]), 4),
                                    "samples": c["samples"],
                                } for c in st.session_state.profiles[t]["columns"]],
                            } for t in target_tables
                        }

                        system_msg = (
                            "You are a senior Data Architect. Return ONLY valid JSON patch:\n"
                            "{ \"tables\": { \"table\": [ {\"column\":\"...\",\"type\":\"...\",\"nullable\":true/false,\"pk\":true/false}, ... ] } }\n"
                            "Do not invent new tables. No commentary."
                        )
                        payload = {"schema_subset": schema_subset, "profiles_subset": prof_subset, "instruction": instruction}
                        raw, err = safe_chat_completion(
                            client,
                            [{"role": "system", "content": system_msg}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                            max_tokens=1200,
                        )
                        if err or raw is None:
                            st.error(f"LLM patch failed: {err}")
                        else:
                            patch = parse_llm_json_or_repair(client, raw, max_repairs=2)
                            if patch.get("_parse_error"):
                                st.error(f"LLM patch parse failed: {patch['_parse_error']}")
                                st.code(raw)
                            else:
                                try:
                                    if "tables" not in patch:
                                        raise ValueError("Patch must include top-level 'tables'.")
                                    for t, cols in patch["tables"].items():
                                        if t in st.session_state.schema:
                                            st.session_state.schema[t] = [{
                                                "column": sanitize_identifier(c["column"]),
                                                "type": str(c.get("type", "TEXT")),
                                                "nullable": bool(c.get("nullable", True)),
                                                "pk": bool(c.get("pk", False)),
                                            } for c in cols]

                                    st.session_state.rel_candidates = infer_relationships_csv(st.session_state.tables, st.session_state.schema, threshold=0.95)
                                    st.session_state.csv_iterations.append({"tables": target_tables, "instruction": instruction})
                                    audit_event("csv", "llm_patch_applied", {"tables": target_tables})
                                    st.success("✅ Patch applied.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"LLM patch failed: {e}")
                                    st.code(raw)

        st.markdown("### Step 2: Human Validation (Schema)")
        final_schema: Dict[str, pd.DataFrame] = {}
        for t_name, cols in st.session_state.schema.items():
            with st.expander(f"Table governance: {t_name}", expanded=False):
                df_cols = pd.DataFrame(cols)
                edited = st.data_editor(df_cols, use_container_width=True, num_rows="fixed", key=f"schema_edit_{t_name}")
                edited["column"] = edited["column"].astype(str).map(sanitize_identifier)
                edited["type"] = edited["type"].astype(str)
                edited["nullable"] = edited["nullable"].astype(bool)
                edited["pk"] = edited["pk"].astype(bool)
                final_schema[t_name] = edited

        if st.button("✅ Save schema edits & recompute relationship candidates", use_container_width=True):
            st.session_state.schema = {t: df.to_dict(orient="records") for t, df in final_schema.items()}
            st.session_state.rel_candidates = infer_relationships_csv(st.session_state.tables, st.session_state.schema, threshold=0.95)
            audit_event("csv", "schema_validated", {"tables": list(final_schema.keys())})
            st.success("✅ Saved. Relationship candidates refreshed.")
            st.rerun()

        st.markdown("### Step 3: Relationship Candidates + HITL approval")
        rel_df = pd.DataFrame(st.session_state.rel_candidates)
        if rel_df.empty:
            st.info("ℹ️ No relationship candidates found.")
        else:
            rel_df = rel_df[["approved", "ref_table", "ref_column", "src_table", "src_column", "cardinality", "containment", "label"]].copy()
            rel_df["containment"] = rel_df["containment"].astype(float).round(4)
            edited_rel = st.data_editor(rel_df, use_container_width=True, key="rel_editor_csv")
            st.session_state.rel_candidates = edited_rel.to_dict(orient="records")

        st.markdown("### Step 4: Final Output (ERD + DDL)")
        if st.button("🪄 Generate Model Assets (CSV)", use_container_width=True):
            approved = [r for r in st.session_state.rel_candidates if r.get("approved")]
            pk_present = any(df["pk"].any() for df in final_schema.values()) if final_schema else False
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Technical Quality", "✅ High" if pk_present else "⚠️ Low")
            c2.metric("Relationships", f"🔗 {len(approved)} approved")
            c3.metric("Tables", f"🗄️ {len(final_schema)}")
            c4.metric("Governance", "✅ Auditable")

            mermaid = build_mermaid_from_schema(final_schema, st.session_state.rel_candidates)
            ddl = build_sql_from_schema(final_schema, st.session_state.rel_candidates)

            audit_event("csv", "model_generated", {"mermaid": mermaid, "ddl": ddl})

            st.subheader("📐 Mermaid ERD")
            st.code(mermaid, language="text")
            render_mermaid(mermaid, height=650)

            st.subheader("🗄️ SQL DDL")
            st.code(ddl, language="sql")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("🗄️ Download DDL (.sql)", ddl, file_name=f"csv_schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql", use_container_width=True)
            with col2:
                st.download_button("📐 Download ERD (.mmd)", mermaid, file_name=f"csv_erd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mmd", use_container_width=True)
    else:
        st.info("📁 Upload CSV files to start CSV modeling.")
