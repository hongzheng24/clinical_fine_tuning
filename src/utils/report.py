"""
Human Review Interface.

The whole point of Agents 5 and 6 is that a human makes the final call --
so this renders the pipeline's output as a single self-contained HTML file
a reviewer can open and skim: each suggested code, its evidence quote (or
lack thereof), and a confidence score, color coded so low-confidence /
unverified items jump out immediately.
"""

import html
import json

_CSS = """
body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 40px auto; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 22px; }
h2 { font-size: 16px; color: #444; margin-top: 32px; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 14px; }
.high { border-left: 6px solid #2e7d32; }
.medium { border-left: 6px solid #f9a825; }
.low { border-left: 6px solid #c62828; background: #fff6f6; }
.code { font-weight: 600; font-size: 15px; }
.quote { font-style: italic; color: #333; background: #f7f7f7; padding: 8px 10px; border-radius: 4px; margin-top: 6px; }
.missing-quote { color: #c62828; font-weight: 500; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; color: white; }
.badge.high { background: #2e7d32; border: none; }
.badge.medium { background: #f9a825; border: none; }
.badge.low { background: #c62828; border: none; }
.warnings { background: #fff3cd; border: 1px solid #ffe08a; border-radius: 6px; padding: 12px; margin-top: 20px; }
.trace { font-family: monospace; font-size: 12px; color: #555; white-space: pre-wrap; background: #fafafa; padding: 12px; border-radius: 6px; }
.section-body { white-space: pre-wrap; background: #fafafa; padding: 10px; border-radius: 6px; font-size: 13px; }
"""


def _tier(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def render_html(state) -> str:
    codes = state.get("codes", [])
    evidence_by_code = {e["icd10_code"]: e for e in state.get("evidence", [])}
    confidence = state.get("confidence", {})
    warnings = state.get("warnings", [])
    sections = state.get("sections", {})
    trace = state.get("trace", [])

    cards = []
    for code in codes:
        code_id = code.get("icd10_code")
        score = confidence.get(code_id, 0.0)
        tier = _tier(score)
        evidence = evidence_by_code.get(code_id, {})
        quote_html = (
            f'<div class="quote">"{html.escape(evidence.get("supporting_quote") or "")}"</div>'
            if evidence.get("verified")
            else '<div class="missing-quote">No supporting quote found in source text -- requires human verification.</div>'
        )
        cards.append(f"""
        <div class="card {tier}">
          <div class="code">{html.escape(str(code_id))} &mdash; {html.escape(code.get("icd10_description", ""))}
            <span class="badge {tier}">{score:.2f} confidence</span>
          </div>
          <div><strong>Diagnosis:</strong> {html.escape(code.get("diagnosis", ""))}</div>
          <div><strong>Reasoning:</strong> {html.escape(code.get("reasoning", ""))}</div>
          {quote_html}
        </div>
        """)

    warnings_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        warnings_html = f'<div class="warnings"><strong>Flagged for human review:</strong><ul>{items}</ul></div>'

    sections_html = "".join(
        f'<h2>{html.escape(name)}</h2><div class="section-body">{html.escape(body[:1500])}</div>'
        for name, body in sections.items()
    )

    trace_html = "\n".join(html.escape(t) for t in trace)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Coding Review -- {html.escape(state.get("doc_id", ""))}</title>
<style>{_CSS}</style></head>
<body>
<h1>Agentic Clinical Coding Review</h1>
<p><strong>Document:</strong> {html.escape(state.get("doc_id", ""))}</p>

<h2>Suggested Codes ({len(codes)})</h2>
{''.join(cards) or '<p>No codes suggested.</p>'}

{warnings_html}

<h2>Detected Sections</h2>
{sections_html}

<h2>Agent Trace (pipeline log)</h2>
<div class="trace">{trace_html}</div>

</body></html>"""


def render_json(state) -> str:
    return json.dumps(
        {
            "doc_id": state.get("doc_id"),
            "sections": list(state.get("sections", {}).keys()),
            "entities": state.get("entities", {}),
            "codes": state.get("codes", []),
            "evidence": state.get("evidence", []),
            "confidence": state.get("confidence", {}),
            "warnings": state.get("warnings", []),
        },
        indent=2,
    )
