'''
Agent 6: Confidence Scoring Agent
'''

from src.utils.state import PipelineState

_LLM_CONFIDENCE_MAP = {"high": 1.0, "medium": 0.6, "low": 0.2}

def score_confidence(state: PipelineState) -> dict:
    '''
    Calculate confidence score based on verified evidence and the LLM's self-reported confidence.
    Max score if there is verified evidence. Else, score is based on evidence match score.

    Parameters
    ==========
    state: PipelineState
        Entity state for which to generate a confidence score.

    Returns
    =======
    confidence_dict: dict
        Map containing confidence score.    
    '''
    code_to_evidence = {entity["icd10_code"]: entity for entity in state.get("evidence", [])}
    scored, warnings = {}, []

    for code in state.get('codes', []):
        code_id = code.get('icd10_code')
        evidence = code_to_evidence.get(code_id, {})

        confidence_score = 1.0 if evidence.get("verified") else max(0.0, evidence.get("match_score", 0.0))
        llm_score = _LLM_CONFIDENCE_MAP.get(code.get("llm_self_reported_confidence", "low"), 0.2)
        entity_score = 1.0

        final_score = round(0.6 * confidence_score + 0.3 * llm_score + 0.1 * entity_score, 2)
        scored[code_id] = final_score 
        
        if final_score < 0.5:
            warnings.append(f"WARNING: LOW CONFIDENCE ({final_score}) for {code_id} -- requires human review before use")

    num_flagged = sum(1 for value in scored.values() if value < 0.5)
    return {
        "confidence": scored,
        "warnings": warnings,
        "trace": [f"[ConfidenceAgent] scored {len(scored)} code(s); flagged {num_flagged} for mandatory review"],
    }