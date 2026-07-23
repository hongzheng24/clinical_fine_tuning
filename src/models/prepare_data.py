from src.agents.coding_agent import _SYSTEM_PROMPT, _build_prompt
from src.utils.state import PipelineState
from src.preprocess.load_medcoder import ExampleClass, build_dataset
from src.utils.pipeline import run_pipeline
import json

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import requests
import numpy as np

from sklearn.model_selection import train_test_split

TEXT_FILEPATH = 'data/raw_data/text.csv'
DIAGNOSIS_FILEPATH = 'data/raw_data/diagnosis.csv'
TEXT_HOLDOUT_FILEPATH = 'data/raw_data/text_holdout.csv'
DIAGNOSIS_HOLDOUT_FILEPATH = 'data/raw_data/diagnosis_holdout.csv'
OUT_DIR = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/fine_tune_data'
ICD10_API_URL = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search'


def _get_code_description(code: str) -> dict[str, str]:
    '''
    Retrieve ICD10 code descriptions from NLM API.

    Parameters
    ==========
    code: str
        ICD10 code.
    
    Returns
    =======
    description: dict[str, str]
        Dict containing code description.
    '''
    params = {
        'sf': 'code,name',
        'terms': code
    }
    try:
        response = requests.get(ICD10_API_URL, params=params)
        data = response.json()
        num_matches = data[0]
        codes = data[1]
        results = data[3]
        print(f'\n{num_matches}\n{codes}\n{results}')
    except Exception as e:
        print(f'An error occured: {e}')

    return {
        'num_matches': num_matches,
        'codes': codes,
        'results': results
    }


def _build_messages(example: ExampleClass, state: PipelineState) -> dict[str, dict]:
    '''
    Construct training/testing messages where the user content corresponds to the prompt
    defined in code_agent.py and the assistant content corresponds to the target response.

    Parameters
    ==========
    example: ExampleClass
        Example case for which to build a message.
    state: PipelineState
        State of example case.

    Returns
    =======
    messages: dict[str, str]
        Dict containing messages for LLM.
    '''
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(state)},
            {"role": "assistant", "content": _build_target(example)},
        ],
    }

def _build_target(example: ExampleClass) -> dict:
    '''
    Build ground-truth target response.

    Parameters
    ==========
    example: ExampleClass
        Example for which to build a target response.

    Returns
    ========
    target: dict
        Dict containing desired response.
    '''
    return json.dumps({
        "codes": [
            {
                "diagnosis": diagnosis,
                "icd10_code": code,
                "icd10_description": _get_code_description(code),
                "reasoning": example.medical_record_text[start: end],
                "llm_self_reported_confidence": "high"
            } for code, diagnosis, start, end in list(zip(example.codes, example.diagnoses, example.starts, example.ends))
        ] 
    })



def _write_json(rows: list[dict], filepath: str):
    '''
    Write messages to JSONL file.

    Parameters:
    ==========
    rows: list[dict]
        List of example messages to be written to JSONL file.
    filepath: str
        Directory path in which to create JSONL file.
    
    Returns
    =======
    None
    '''
    path = ...
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return



def prepare():
    '''
    Prepare train, validation, and test messages for SFTTrainer.
    '''
    # Prepare train and validation data
    examples = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH)
    messages = [_build_messages(example) for example in examples]
    train_messages, val_messages = train_test_split(messages, test_size=0.1, random_state=42)

    # Prepare test (holdout) data
    holdout_examples = build_dataset(TEXT_HOLDOUT_FILEPATH, DIAGNOSIS_HOLDOUT_FILEPATH)
    test_messages = [_build_messages(example) for example in holdout_examples]

    # Write prepared training messages to json file
    _write_json(train_messages, f'{OUT_DIR}/train_data.jsonl')

    # Write prepared valuation messages to json file
    _write_json(val_messages, f'{OUT_DIR}/val_data.jsonl')
    return


if __name__ == '__main__':
    prepare()