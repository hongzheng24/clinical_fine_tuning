'''
Agent 2: Section Detection

regex for all caps section titles: \b[A-Z]{2,}\b
'''

import logging
import medspacy
from medspacy.section_detection import SectionRule
from src.utils.state import PipelineState

logging.disable(logging.CRITICAL)
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except ImportError:
    pass


# Section rules. Grouped by topic and in desired order. More general section titles are processed later.
_SECTION_RULES = [
    # Main titles
    SectionRule(literal='CHIEF COMPLAINT', category='chief_complaint', pattern=[{'TEXT': 'CHIEF'}, {'TEXT': 'COMPLAINT'}]),
    SectionRule(literal='REVIEW OF SYSTEMS', category='review_of_systems', pattern=[{'TEXT': 'REVIEW'}, {'TEXT': 'OF'}, {'TEXT': 'SYSTEMS'}]),

    # Exams
    SectionRule(literal='PHYSICAL EXAM CONSTITUTIONAL', category='physical_exam_constitutional', pattern=[{'TEXT': 'PHYSICAL'}, {'TEXT': 'EXAM'}, {'TEXT': 'CONSTITUTIONAL'}]),
    SectionRule(literal='PHYSICAL EXAM CV', category='physical_exam_cv', pattern=[{'TEXT': 'PHYSICAL'}, {'TEXT': 'EXAM'}, {'TEXT': 'CV'}]),
    SectionRule(literal='PHYSICAL EXAM GAIT', category='physical_exam_gait', pattern=[{'TEXT': 'PHYSICAL'}, {'TEXT': 'EXAM'}, {'TEXT': 'GAIT'}]),
    SectionRule(literal='PHYSICAL EXAM NEURO', category='physical_exam_neuro', pattern=[{'TEXT': 'PHYSICAL'}, {'TEXT': 'EXAM'}, {'TEXT': 'NEURO'}]),
    SectionRule(literal='PHYSICAL EXAMINATION', category='physical_examination', pattern=[{'TEXT': 'PHYSICAL'}, {'TEXT': 'EXAMINATION'}]),
    SectionRule(literal='PHYSICAL EXAM', category='physical_exam', pattern=[{'TEXT': 'PHYSICAL'}, {'TEXT': 'EXAM'}]),
    SectionRule(literal='EXAM', category='exam', pattern=[{'TEXT': 'EXAM'}]),

    # Assessment and plan
    SectionRule(literal='ASSESSMENT AND PLAN', category='assessment_and_plan', pattern=[{'TEXT': 'ASSESSMENT'}, {'TEXT': 'AND'}, {'TEXT': 'PLAN'}]),
    SectionRule(literal='ASSESSMENT', category='assessment', pattern=[{'TEXT': 'ASSESSMENT'}]),

    SectionRule(literal='PLAN', category='plan', pattern=[{'TEXT': 'PLAN'}]),

    # History
    SectionRule(literal='PAST MEDICAL HISTORY', category='past_medical_history', pattern=[{'TEXT': 'PAST'}, {'TEXT': 'MEDICAL'}, {'TEXT': 'HISTORY'}]),
    SectionRule(literal='PAST HISTORY', category='past_history', pattern=[{'TEXT': 'PAST'}, {'TEXT': 'HISTORY'}]),
    SectionRule(literal='FAMILY HISTORY', category='family_history', pattern=[{'TEXT': 'FAMILY'}, {'TEXT': 'HISTORY'}]),
    SectionRule(literal='MEDICAL HISTORY', category='medical_history', pattern=[{'TEXT': 'MEDICAL'}, {'TEXT': 'HISTORY'}]),
    SectionRule(literal='SOCIAL HISTORY', category='social_history', pattern=[{'TEXT': 'SOCIAL'}, {'TEXT': 'HISTORY'}]),
    SectionRule(literal='SURGICAL HISTORY', category='surgical_history', pattern=[{'TEXT': 'SURGICAL'}, {'TEXT': 'HISTORY'}]),
    SectionRule(literal='HISTORY OF PRESENT ILLNESS', category='history_of_present_illness', pattern=[{'TEXT': 'HISTORY'}, {'TEXT': 'OF'}, {'TEXT': 'PRESENT'}, {'TEXT': 'ILLNESS'}]),

    # Medications
    SectionRule(literal='CURRENT MEDICATIONS', category='current_medications', pattern=[{'TEXT': 'CURRENT'}, {'TEXT': 'MEDICATIONS'}]),
    SectionRule(literal='MEDICATIONS', category='medications', pattern=[{'TEXT': 'MEDICATIONS'}]),

    # Others
    SectionRule(literal='ALLERGIES', category='allergies', pattern=[{'TEXT': 'ALLERGIES'}]),
    SectionRule(literal='RESULTS', category='results', pattern=[{'TEXT': 'RESULTS'}]),
    SectionRule(literal='HEAD', category='head', pattern=[{'TEXT': 'HEAD'}]),
    SectionRule(literal='RESPIRATORY', category='respiratory', pattern=[{'TEXT': 'RESPIRATORY'}]),
    SectionRule(literal='VITALS REVIEWED', category='vitals_reviewed', pattern=[{'TEXT': 'VITALS'}, {'TEXT': 'REVIEWED'}]),
    SectionRule(literal='VITALS', category='vitals', pattern=[{'TEXT': 'VITALS'}]),
    SectionRule(literal='INSTRUCTIONS', category='instructions', pattern=[{'TEXT': 'INSTRUCTIONS'}]),
    SectionRule(literal='MSK', category='msk', pattern=[{'TEXT': 'MSK'}]),
    SectionRule(literal='CC', category='cc', pattern=[{'TEXT': 'CC'}]),
    SectionRule(literal='HPI', category='hpe', pattern=[{'TEXT': 'HPI'}]),
    SectionRule(literal='IMPRESSION', category='impression', pattern=[{'TEXT': 'IMPRESSION'}]),
    SectionRule(literal='SKIN', category='skin', pattern=[{'TEXT': 'SKIN'}]),
    SectionRule(literal='HENT', category='hent', pattern=[{'TEXT': 'HENT'}]),
    SectionRule(literal='SUBJECTIVE', category='subjective', pattern=[{'TEXT': 'SUBJECTIVE'}]),
    SectionRule(literal='CV', category='cv', pattern=[{'TEXT': 'CV'}]),
    # SectionRule(literal='EKG', category='ekg', pattern=[{'TEXT': 'EKG'}]),    
]

_nlp = None

def _get_nlp() -> None:
    '''
    Load MedSpaCy NLP model.

    Parameters
    ==========
    None

    Returns
    =======
    None
    '''
    global _nlp

    if _nlp:
        return _nlp

    _nlp = medspacy.load(enable=["medspacy_pyrush"]) # _nlp = medspacy.load(enable=["medspacy_tokenizer", "medspacy_pyrush"])

    # Here, the sectionizer is case-sensitive and returns exact matches. If this behavior is not desired,
    # please use the following sectionizer instead.

    # sectionizer = _nlp.add_pipe("medspacy_sectionizer")
    sectionizer = _nlp.add_pipe("medspacy_sectionizer", config={"require_start_line": True, "rules": None})

    sectionizer.add(_SECTION_RULES)

    return _nlp


def detect_sections(state: PipelineState) -> dict:
    '''
    Detect document sections by category using MedSpaCy model.

    Parameters
    ==========
    Clinical case state for which to conduct section detection.

    Returns
    =======
    sections_dict: dict
        Map containing detected document sections.
    '''
    nlp = _get_nlp()
    document = nlp(state['text'])

    sections = {}
    for section in document._.sections:
        category = section.category or 'PREAMBLE'
        start_token, end_token = section.body_span
        section_text = document[start_token:end_token].text.strip()
        if section_text:
            sections[category] = (sections.get(category, "") + "\n" + section_text).strip()

    return {
        "sections": sections,
        "trace": [f"[SectionAgent] detected {len(sections)} section(s): {list(sections.keys())}"],
    }