"""
Shared pipeline state, implemented with LangGraph-native.

LangGraph nodes take the current state and return a *partial* update dict --
LangGraph merges it into state for you. Plain keys are overwritten (fine
here since each key is only ever written by one node); `warnings` and
`trace` are Annotated with `operator.add` so multiple nodes can each
append to the same running log instead of clobbering each other.
"""

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    # --- input ---
    doc_id: str
    raw_input: str          # raw text OR a file path to a PDF
    is_pdf: bool

    # --- Agent 1: Document Loader ---
    text: str

    # --- Agent 2: Section Detection ---
    sections: dict[str, str]

    # --- Agent 3: Clinical Entity Extraction ---
    entities: dict[str, list[dict[str, Any]]]

    # --- Agent 4: Coding Recommendation (LLM) ---
    codes: list[dict[str, Any]]

    # --- Agent 5: Evidence Verification ---
    evidence: list[dict[str, Any]]

    # --- Agent 6: Confidence Scoring ---
    confidence: dict[str, float]

    # --- cross-cutting, accumulated across nodes ---
    warnings: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
