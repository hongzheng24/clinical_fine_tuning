'''
Agent 5: Evidence Verification
'''

import difflib
import re
from src.utils.state import PipelineState

def _process_sentences(text: str) -> list[str]:
    '''
    Process sentences.

    Parameters
    ==========
    text: str
        String to be processed.

    Returns
    =======
    processed_text: str
        Cleaned processed text.
    '''
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _best_matching_sentence(term: str, sentences: list[str]) -> tuple[str | None, float]:
    '''
    Find sentence in document with the highest matching score with respect to given medical term.

    Parameters
    ==========
    term: str
        Medical term for which to determine a best matching sentence and score.
    sentences: list[str]
        Candidate sentences to be scored.

    Returns
    =======
    best_sentence: str
        Sentence with highest matching score.
    best_score: float
        Highest matching score.
    '''
    term_lower = term.lower()
    best_sentence, best_score = None, 0.0
    for sentence in sentences:
        if term_lower in sentence.lower():
            return sentence, 1.0
        score = difflib.SequenceMatcher(None, term_lower, sentence.lower()).ratio()
        if score > best_score:
            best_sentence, best_score = sentence, score
    return best_sentence, best_score

def verify_evidence(state: PipelineState) -> dict:
    sentences = _process_sentences(state['text'])
    evidence_list, warnings = [], []

    for code in state.get('codes', []):
        diagnosis = code.get('diagnosis', '')
        sentence, score = _best_matching_sentence(diagnosis, sentences)
        found = score >= 0.98 or (sentence and diagnosis.lower() in sentence.lower())

        evidence_list.append({
            "diagnosis": diagnosis,
            "icd10_code": code.get("icd10_code"),
            "supporting_quote": sentence if found else None,
            "match_score": round(score, 2),
            "verified": bool(found),
        })

        if not found:
            warnings.append(
                f"No strong textual support found for '{diagnosis}' "
                f"({code.get('icd10_code')}) -- flagged for human review"
            )

    num_evidence = sum(1 for e in evidence_list if e["verified"])
    return {
        "evidence": evidence_list,
        "warnings": warnings,
        "trace": [f"[VerificationAgent] verified {num_evidence}/{len(evidence_list)} suggested code(s) against source text"],
    }