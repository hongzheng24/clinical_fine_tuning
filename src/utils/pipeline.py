"""
Pipeline orchestrator -- built with LangGraph's StateGraph.

Each agent is a graph node; every node receives the full shared state and
returns a partial update, which LangGraph merges in (see agents/state.py
for the reducer rules on `warnings`/`trace`). Nodes never call each other
directly -- they only read/write shared state, and the graph owns control
flow. For this POC the graph is a straight line, but this is exactly the
same primitive you'd use to add branches later -- e.g. a conditional edge
that routes low-confidence codes to a "second opinion" node instead of
straight to the human reviewer.
"""

from langgraph.graph import END, StateGraph
from src.agents.coding_agent import recommend_codes
from src.agents.confidence_agent import score_confidence
from src.agents.document_agent import load_document
from src.agents.entity_agent import extract_entities
from src.agents.section_agent import detect_sections
from src.utils.state import PipelineState
from src.agents.verification_agent import verify_evidence


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("load_document", load_document)
    graph.add_node("detect_sections", detect_sections)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("recommend_codes", recommend_codes)
    graph.add_node("verify_evidence", verify_evidence)
    graph.add_node("score_confidence", score_confidence)

    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "detect_sections")
    graph.add_edge("detect_sections", "extract_entities")
    graph.add_edge("extract_entities", "recommend_codes")
    graph.add_edge("recommend_codes", "verify_evidence")
    graph.add_edge("verify_evidence", "score_confidence")
    graph.add_edge("score_confidence", END)

    return graph.compile()


app = build_graph()


def run_pipeline(doc_id: str, raw_input: str, is_pdf: bool = False) -> PipelineState:
    initial_state: PipelineState = {
        "doc_id": doc_id,
        "raw_input": raw_input,
        "is_pdf": is_pdf,
        "warnings": [],
        "trace": [],
    }
    return app.invoke(initial_state)


def export_graph_diagram(path: str = "output/graph.png") -> None:
    """Nice for slides: renders the actual compiled graph, not a hand-drawn
    approximation of it. Requires network access to render via Mermaid's
    hosted API -- falls back to an ASCII sketch if that's unavailable."""
    try:
        app.get_graph().draw_mermaid_png(output_file_path=path)
        print(f"wrote {path}")
    except Exception:
        print(app.get_graph().draw_ascii())
