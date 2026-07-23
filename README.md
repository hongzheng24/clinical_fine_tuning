



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


BUG
=================
- ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /users/hzheng29/.conda/envs/clinical-fine-tuning/lib/python3.10/site-packages/scipy/spatial/_distance_pybind.cpython-310-x86_64-linux-gnu.so)
        - FIX: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wasiahmad/software/anaconda3/lib/
        - What caused this? fine_tune.py worked before



Design Choices
==================
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

Evaluation
    Multi-label metrics. Each clinical note may have multiple ICD-10 codes.

