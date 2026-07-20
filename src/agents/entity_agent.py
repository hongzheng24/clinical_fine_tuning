import medspacy
from medspacy.ner import TargetRule
from src.utils.state import PipelineState

_nlp = None

# Subset of i2b2 target rules. Find exhaustive list if needed.
_TARGET_RULES = [
    # Problems
    TargetRule(literal="hypertension", category="PROBLEM"),
    TargetRule(literal="essential hypertension", category="PROBLEM"),
    TargetRule(literal="hypertensive urgency", category="PROBLEM"),
    TargetRule(literal="type 2 diabetes mellitus", category="PROBLEM"),
    TargetRule(literal="diabetes", category="PROBLEM"),
    TargetRule(literal="hyperlipidemia", category="PROBLEM"),
    TargetRule(literal="community-acquired pneumonia", category="PROBLEM"),
    TargetRule(literal="pneumonia", category="PROBLEM"),
    TargetRule(literal="pulmonary embolism", category="PROBLEM"),
    TargetRule(literal="st-elevation myocardial infarction", category="PROBLEM"),
    TargetRule(literal="myocardial infarction", category="PROBLEM"),
    TargetRule(literal="cholelithiasis", category="PROBLEM"),
    TargetRule(literal="cholecystitis", category="PROBLEM"),
    TargetRule(literal="cholangitis", category="PROBLEM"),
    TargetRule(literal="pancreatitis", category="PROBLEM"),
    TargetRule(literal="anemia", category="PROBLEM"),
    TargetRule(literal="sepsis", category="PROBLEM"),
    TargetRule(literal="acute kidney injury", category="PROBLEM"),
    TargetRule(literal="chronic kidney disease", category="PROBLEM"),
    TargetRule(literal="atrial fibrillation", category="PROBLEM"),
    TargetRule(literal="congestive heart failure", category="PROBLEM"),
    TargetRule(literal="heart failure", category="PROBLEM"),
    # Medications
    TargetRule(literal="lisinopril", category="MEDICATION"),
    TargetRule(literal="metformin", category="MEDICATION"),
    TargetRule(literal="atorvastatin", category="MEDICATION"),
    TargetRule(literal="ceftriaxone", category="MEDICATION"),
    TargetRule(literal="azithromycin", category="MEDICATION"),
    TargetRule(literal="aspirin", category="MEDICATION"),
    TargetRule(literal="ticagrelor", category="MEDICATION"),
    TargetRule(literal="metoprolol", category="MEDICATION"),
    TargetRule(literal="ketorolac", category="MEDICATION"),
    TargetRule(literal="furosemide", category="MEDICATION"),
    TargetRule(literal="insulin", category="MEDICATION"),
    TargetRule(literal="warfarin", category="MEDICATION"),
    # Procedures
    TargetRule(literal="cardiac catheterization", category="PROCEDURE"),
    TargetRule(literal="percutaneous coronary intervention", category="PROCEDURE"),
    TargetRule(literal="drug-eluting stent", category="PROCEDURE"),
    TargetRule(literal="laparoscopic cholecystectomy", category="PROCEDURE"),
    TargetRule(literal="dialysis", category="PROCEDURE"),
    TargetRule(literal="intubation", category="PROCEDURE"),
]


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
    
    _nlp = medspacy.load() 
    matcher = _nlp.get_pipe("medspacy_target_matcher")
    matcher.add(_TARGET_RULES)
    return _nlp

def extract_entities(state: PipelineState) -> dict:
    '''
    Extract problem, medication, and procedure entities from patient state using MedSpaCy.

    Parameters
    ==========
    state: PipelineState
        Patient state for which to extract entities.
    
    Returns
    =======
    entities_dict: dict
        Map containing extracted entities.
    '''
    nlp = _get_nlp()
    document = nlp(state["text"])

    entities = {
        'PROBLEM': [],
        'MEDICATION': [],
        'PROCEDURE': []
    }
    seen = set()

    for entity in document.ents:
        key = (entity.text.lower().strip(), entity.label_, entity._.is_negated)
        if key in seen:
            continue

        seen.add(key)
        entities.setdefault(entity.label_, []).append({
            "text": entity.text,
            "label": entity.label_,
            "negated": bool(entity._.is_negated),
            "is_family": bool(entity._.is_family),
            "is_historical": bool(entity._.is_historical),
        })

    num_active = sum(1 for group in entities.values() for entity in group if not entity["negated"])
    num_negated = sum(1 for group in entities.values() for entity in group if entity["negated"])
    return {
        "entities": entities,
        "trace": [f"[EntityAgent] extracted {num_active} active entities, {num_negated} negated (excluded from coding)"],
    }