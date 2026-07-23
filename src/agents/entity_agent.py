import medspacy
from medspacy.ner import TargetRule
from src.utils.state import PipelineState

_nlp = None

# Subset of I2B2 target rules. Find exhaustive list if needed.
# Target rules can be scaled by retrieving data from the ICD-10-CM
# order file (CMS/CDC).
_TARGET_RULES = [
    # Problems
    TargetRule(literal='gerd', category='PROBLEM'),
    TargetRule(literal='arthritis', category='PROBLEM'),
    TargetRule(literal='upper respiratory problem', category='PROBLEM'),
    TargetRule(literal='hypertension', category='PROBLEM'),
    TargetRule(literal='essential hypertension', category='PROBLEM'),
    TargetRule(literal='hypertensive urgency', category='PROBLEM'),
    TargetRule(literal='type 2 diabetes mellitus', category='PROBLEM'),
    TargetRule(literal='diabetes', category='PROBLEM'),
    TargetRule(literal='hyperlipidemia', category='PROBLEM'),
    TargetRule(literal='community-acquired pneumonia', category='PROBLEM'),
    TargetRule(literal='pneumonia', category='PROBLEM'),
    TargetRule(literal='pulmonary embolism', category='PROBLEM'),
    TargetRule(literal='st-elevation myocardial infarction', category='PROBLEM'),
    TargetRule(literal='myocardial infarction', category='PROBLEM'),
    TargetRule(literal='cholelithiasis', category='PROBLEM'),
    TargetRule(literal='cholecystitis', category='PROBLEM'),
    TargetRule(literal='cholangitis', category='PROBLEM'),
    TargetRule(literal='pancreatitis', category='PROBLEM'),
    TargetRule(literal='anemia', category='PROBLEM'),
    TargetRule(literal='sepsis', category='PROBLEM'),
    TargetRule(literal='acute kidney injury', category='PROBLEM'),
    TargetRule(literal='chronic kidney disease', category='PROBLEM'),
    TargetRule(literal='atrial fibrillation', category='PROBLEM'),
    TargetRule(literal='congestive heart failure', category='PROBLEM'),
    TargetRule(literal='heart failure', category='PROBLEM'),
    # Medications
    TargetRule(literal='lisinopril', category='MEDICATION'),
    TargetRule(literal='metformin', category='MEDICATION'),
    TargetRule(literal='atorvastatin', category='MEDICATION'),
    TargetRule(literal='ceftriaxone', category='MEDICATION'),
    TargetRule(literal='azithromycin', category='MEDICATION'),
    TargetRule(literal='aspirin', category='MEDICATION'),
    TargetRule(literal='ticagrelor', category='MEDICATION'),
    TargetRule(literal='metoprolol', category='MEDICATION'),
    TargetRule(literal='ketorolac', category='MEDICATION'),
    TargetRule(literal='furosemide', category='MEDICATION'),
    TargetRule(literal='insulin', category='MEDICATION'),
    TargetRule(literal='warfarin', category='MEDICATION'),
    # Procedures
    TargetRule(literal='cardiac catheterization', category='PROCEDURE'),
    TargetRule(literal='percutaneous coronary intervention', category='PROCEDURE'),
    TargetRule(literal='drug-eluting stent', category='PROCEDURE'),
    TargetRule(literal='laparoscopic cholecystectomy', category='PROCEDURE'),
    TargetRule(literal='dialysis', category='PROCEDURE'),
    TargetRule(literal='intubation', category='PROCEDURE'),
    # Problems: cardiovascular
    TargetRule(literal='atrial fibrillation', category='PROBLEM'),
    TargetRule(literal='atrial flutter', category='PROBLEM'),
    TargetRule(literal='congestive heart failure', category='PROBLEM'),
    TargetRule(literal='heart failure with reduced ejection fraction', category='PROBLEM'),
    TargetRule(literal='coronary artery disease', category='PROBLEM'),
    TargetRule(literal='peripheral vascular disease', category='PROBLEM'),
    TargetRule(literal='deep vein thrombosis', category='PROBLEM'),
    TargetRule(literal='aortic stenosis', category='PROBLEM'),
    TargetRule(literal='mitral regurgitation', category='PROBLEM'),
    TargetRule(literal='cardiogenic shock', category='PROBLEM'),
    # Problems: respiratory
    TargetRule(literal='chronic obstructive pulmonary disease', category='PROBLEM'),
    TargetRule(literal='copd', category='PROBLEM'),
    TargetRule(literal='asthma', category='PROBLEM'),
    TargetRule(literal='acute respiratory failure', category='PROBLEM'),
    TargetRule(literal='pleural effusion', category='PROBLEM'),
    TargetRule(literal='pneumothorax', category='PROBLEM'),
    TargetRule(literal='obstructive sleep apnea', category='PROBLEM'),
    # Problems: renal / metabolic
    TargetRule(literal='acute kidney injury', category='PROBLEM'),
    TargetRule(literal='chronic kidney disease', category='PROBLEM'),
    TargetRule(literal='end stage renal disease', category='PROBLEM'),
    TargetRule(literal='hyponatremia', category='PROBLEM'),
    TargetRule(literal='hyperkalemia', category='PROBLEM'),
    TargetRule(literal='diabetic ketoacidosis', category='PROBLEM'),
    TargetRule(literal='hypothyroidism', category='PROBLEM'),
    TargetRule(literal='hyperthyroidism', category='PROBLEM'),
    TargetRule(literal='obesity', category='PROBLEM'),
    # Problems: GI
    TargetRule(literal='gastroesophageal reflux disease', category='PROBLEM'),
    TargetRule(literal='peptic ulcer disease', category='PROBLEM'),
    TargetRule(literal='gastrointestinal bleed', category='PROBLEM'),
    TargetRule(literal='acute pancreatitis', category='PROBLEM'),
    TargetRule(literal='cirrhosis', category='PROBLEM'),
    TargetRule(literal='diverticulitis', category='PROBLEM'),
    TargetRule(literal='small bowel obstruction', category='PROBLEM'),
    TargetRule(literal='appendicitis', category='PROBLEM'),
    # Problems: infectious / heme-onc / neuro / psych
    TargetRule(literal='sepsis', category='PROBLEM'),
    TargetRule(literal='septic shock', category='PROBLEM'),
    TargetRule(literal='urinary tract infection', category='PROBLEM'),
    TargetRule(literal='cellulitis', category='PROBLEM'),
    TargetRule(literal='influenza', category='PROBLEM'),
    TargetRule(literal='anemia', category='PROBLEM'),
    TargetRule(literal='thrombocytopenia', category='PROBLEM'),
    TargetRule(literal='cerebrovascular accident', category='PROBLEM'),
    TargetRule(literal='transient ischemic attack', category='PROBLEM'),
    TargetRule(literal='seizure disorder', category='PROBLEM'),
    TargetRule(literal='major depressive disorder', category='PROBLEM'),
    TargetRule(literal='depression', category='PROBLEM'),
    TargetRule(literal='generalized anxiety disorder', category='PROBLEM'),
    TargetRule(literal='dementia', category='PROBLEM'),
    TargetRule(literal='delirium', category='PROBLEM'),
    # Medications: cardiac / renal
    TargetRule(literal='amlodipine', category='MEDICATION'),
    TargetRule(literal='losartan', category='MEDICATION'),
    TargetRule(literal='carvedilol', category='MEDICATION'),
    TargetRule(literal='furosemide', category='MEDICATION'),
    TargetRule(literal='spironolactone', category='MEDICATION'),
    TargetRule(literal='apixaban', category='MEDICATION'),
    TargetRule(literal='warfarin', category='MEDICATION'),
    TargetRule(literal='clopidogrel', category='MEDICATION'),
    # Medications: endocrine / GI
    TargetRule(literal='insulin glargine', category='MEDICATION'),
    TargetRule(literal='insulin lispro', category='MEDICATION'),
    TargetRule(literal='levothyroxine', category='MEDICATION'),
    TargetRule(literal='omeprazole', category='MEDICATION'),
    TargetRule(literal='pantoprazole', category='MEDICATION'),
    # Medications: antibiotics / analgesics / psych / pulm
    TargetRule(literal='vancomycin', category='MEDICATION'),
    TargetRule(literal='piperacillin-tazobactam', category='MEDICATION'),
    TargetRule(literal='cefepime', category='MEDICATION'),
    TargetRule(literal='acetaminophen', category='MEDICATION'),
    TargetRule(literal='oxycodone', category='MEDICATION'),
    TargetRule(literal='morphine', category='MEDICATION'),
    TargetRule(literal='gabapentin', category='MEDICATION'),
    TargetRule(literal='sertraline', category='MEDICATION'),
    TargetRule(literal='albuterol', category='MEDICATION'),
    TargetRule(literal='prednisone', category='MEDICATION'),
    TargetRule(literal='heparin', category='MEDICATION'),
    # Procedures
    TargetRule(literal='colonoscopy', category='PROCEDURE'),
    TargetRule(literal='endoscopy', category='PROCEDURE'),
    TargetRule(literal='computed tomography', category='PROCEDURE'),
    TargetRule(literal='ct scan', category='PROCEDURE'),
    TargetRule(literal='magnetic resonance imaging', category='PROCEDURE'),
    TargetRule(literal='mri', category='PROCEDURE'),
    TargetRule(literal='echocardiogram', category='PROCEDURE'),
    TargetRule(literal='electrocardiogram', category='PROCEDURE'),
    TargetRule(literal='hemodialysis', category='PROCEDURE'),
    TargetRule(literal='mechanical ventilation', category='PROCEDURE'),
    TargetRule(literal='central line placement', category='PROCEDURE'),
    TargetRule(literal='blood transfusion', category='PROCEDURE'),
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
    matcher = _nlp.get_pipe('medspacy_target_matcher')
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
    document = nlp(state['text'])

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
            'text': entity.text,
            'label': entity.label_,
            'negated': bool(entity._.is_negated),
            'is_family': bool(entity._.is_family),
            'is_historical': bool(entity._.is_historical),
        })

    num_active = sum(1 for group in entities.values() for entity in group if not entity['negated'])
    num_negated = sum(1 for group in entities.values() for entity in group if entity['negated'])
    return {
        'entities': entities,
        'trace': [f'[EntityAgent] extracted {num_active} active entities, {num_negated} negated (excluded from coding)'],
    }