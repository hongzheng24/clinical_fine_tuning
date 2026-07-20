'''
Agent 2: Section Detection
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

    _nlp = medspacy.load(enable=["medspacy_pyrush"])
    sectionizer = _nlp.add_pipe("medspacy_sectionizer")
    sectionizer.add([SectionRule(literal="ASSESSMENT AND PLAN:", category="assessment_and_plan")])

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