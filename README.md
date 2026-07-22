07/16/2026 Check in
- Implemented daat/load_medcoder.py
- Looked at medcoder data and imported 2 examples
- Test examples
- Adjusted coding agent prompt
- TODO: Fine-tuning and evaluation




TODO:
- Implement fine-tuning and LLM evaluation
- Check outputs
- Check outputs for real data



EVALUATION DESIGN
=================
- be separate ```python main.py --evaluate```
- should be ran after training


TRAINING DESIGN
===============
- ```python main.py --train```



DESIGN CHOICES
==============
SectionAgent
    SectionAgent's detect_sections uses the MedSpaCy sectionizer. By
    default, the sectionizer is case-insensitive and may return sections that are
    not explicity defined. For the MedCodER dataset, sections are uppercase, defined,
    and finite. Thus, case-sensitivity and exact matching is enabled. See _get_nlp()
    in section_agent for more details.

EntityAgent
    EntityAgent's target rules are not exhaustive but can be scaled by retrieving data
    from the CMS/CDC ICD-10-CM order file. See _TARGET_RULES in entity_agent.py for 
    more details.

