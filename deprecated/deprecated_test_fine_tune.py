from src.utils.state import PipelineState

# Run test: <python -m tests.test>
# <python -m unittest discover -s tests>
# python -m unittest discover -s tests

import unittest
from src.preprocess.load_medcoder import build_dataset, ExampleClass
from src.models.fine_tune import _build_messages, _build_target
from src.agents.coding_agent import _SYSTEM_PROMPT, _build_prompt
from src.utils.state import PipelineState
from src.utils.pipeline import run_pipeline

import json

TEXT_FILEPATH = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/raw_data/text.csv'
DIAGNOSIS_FILEPATH = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/raw_data/diagnosis.csv'

dataset = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH, None, None, 5, 10, None, None)
 
class TestFineTune(unittest.TestCase):
    def setUp(self):
        # For example 0
        self.example_0: ExampleClass = dataset[0]

        # self.state_0: PipelineState = run_pipeline(None, self.example_0.medical_record_text, False)
        self.state_0: PipelineState = {
            "doc_id": self.example_0.doc_id,
            "raw_input": self.example_0.medical_record_text,
            "is_pdf": False,
            "warnings": [],
            "trace": [],
        }

        self.prompt_0: str = _build_prompt(self.state_0)
        self.messages: dict[str, list] = _build_messages(self.example_0, self.state_0)
        self.target_0 = _build_target(self.example_0)
        pass

    def test_build_target(self):
        '''
        Test that _build_target creates the currect target json for example 0
        '''

        true_target_0 = json.dumps({
            'codes': [
                {
                    'diagnosis': 'Upper respiratory infection',
                    'icd10_code': 'J06.9',
                    'reasoning': 'Upper respiratory infection', # TODO: Find exact reasoning text, Does \n char effect reasoning text?
                    'llm_self_reported_confidence': 'high'
                },
                {
                    'diagnosis': 'Depression',
                    'icd10_code': 'F32.A',
                    'reasoning': 'Depression',
                    'llm_self_reported_confidence': 'high'
                },
                {
                    'diagnosis': 'Diabetes type 2',
                    'icd10_code': 'E11.9',
                    'reasoning': 'Diabetes type 2',
                    'llm_self_reported_confidence': 'high'
                },
                {
                    'diagnosis': 'Hypertension',
                    'icd10_code': 'I10',
                    'reasoning': 'Hypertension',
                    'llm_self_reported_confidence': 'high'
                }
            ]
        })

        self.assertEqual(self.target_0, true_target_0)
        pass

    def test_build_messages(self):
        '''Test that _build_messages() creates the correct message format for '''
        print(self.prompt_0)
        true_messages = {
            "messages": [
                {"role": "system", "content": '''You are a clinical coding assistant supporting a human medical coder.
                    You will be given clinical entities already extracted from a note (problems,
                    medications, procedures), each marked as negated or not. You must NOT invent
                    findings that are not in the provided entity list or section text.

                    For each ACTIVE (non-negated) problem, suggest one plausible ICD-10-CM code.
                    If you are not confident of the exact code, give your best general-category
                    code and say so in the reasoning.

                    ICD-10-CM codes must be formatted exactly as standard ICD-10-CM codes:
                    - Use uppercase letters and digits only, with no spaces or extra punctuation.
                    - The code must start with 1 letter followed by 2 digits.
                    - If the code has more than 3 characters, place a decimal point after the first 3 characters.
                    - Do not add leading zeros, trailing zeros, or any unlisted characters.
                    - Valid examples: I10, E11.9, J44.9, Z79.4
                    - Invalid examples: e119, E11 9, E11-9, 11.9, E11.90

                    Return ONLY valid JSON, no markdown fences, no preamble, matching this shape:
                    {
                    "codes": [
                        {
                            "diagnosis": "string, the clinical concept",
                            "icd10_code": "string",
                            "icd10_description": "string",
                            "reasoning": "string, 1-2 sentences, must reference only the given entities/sections",
                            "llm_self_reported_confidence": "high" | "medium" | "low"
                        }
                    ]
                    }'''
                },
                {"role": "user", "content": 'ACTIVE PROBLEMS (extracted, not negated): [ "diabetes", "hypertension" ]\n\n EXPLICITLY NEGATED / RULED-OUT (do NOT code these): [ "pneumonia" ]\n\n MEDICATIONS ON RECORD: [ "lisinopril", "metformin" ]\n\n PROCEDURES ON RECORD: []\n\n RELEVANT NOTE TEXT (Assessment/Plan or main body): CHIEF COMPLAINT\n\n Upper respiratory infection.\n\n HISTORY OF PRESENT ILLNESS\n\n Andrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\n The patient reports that he has been doing a bit of work out in the yard in the last week or so. He started to feel really tired and short of breath. The patient denies coughing up anything, but he feels like he will soon be coughing up phlegm. He denies having a fever, but he felt a little warm. He attributed this to exerting himself. He reports that his elbows hurt quite a bit. He notes his knees were pretty tired and he felt some tension around his knees. The patient attributes these symptoms to lifting heavy bags. He has not been wearing a mask as much recently. He believes that he caught his first cold and his symptoms have worsened. He has had both of his COVID vaccinations.'
                }, # TODO: Find exact prepared prompt
                {"role": "assistant", "content": json.dumps({
                    'codes': [
                        {
                            'diagnosis': 'Upper respiratory infection',
                            'icd10_code': 'J06.9',
                            'reasoning': 'Upper respiratory infection', # TODO: Find exact reasoning text, Does \n char effect reasoning text?
                            'llm_self_reported_confidence': 'high'
                        },
                        {
                            'diagnosis': 'Depression',
                            'icd10_code': 'F32.A',
                            'reasoning': 'Depression',
                            'llm_self_reported_confidence': 'high'
                        },
                        {
                            'diagnosis': 'Diabetes type 2',
                            'icd10_code': 'E11.9',
                            'reasoning': 'Diabetes type 2',
                            'llm_self_reported_confidence': 'high'
                        },
                        {
                            'diagnosis': 'Hypertension',
                            'icd10_code': 'I10',
                            'reasoning': 'Hypertension',
                            'llm_self_reported_confidence': 'high'
                        }
                    ]

                })},
            ]
        }
        self.assertTrue(
            self.messages, true_messages
        )
        pass



    


if __name__ == '__main__':
    unittest.main()




'''
Given by github copilot
'''


import json
import unittest
from unittest.mock import patch

from src.preprocess.load_medcoder import ExampleClass
from src.models import fine_tune as fine_tune_module


class FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_called = False
        self.saved_model_dir = None
        self.memorized_targets = []
        FakeTrainer.instances.append(self)

    def train(self):
        self.train_called = True
        self.memorized_targets = [
            entry["messages"][-1]["content"]
            for entry in self.kwargs["train_dataset"]
        ]

    def save_model(self, output_dir):
        self.saved_model_dir = output_dir


class TestFineTune(unittest.TestCase):
    def setUp(self):
        FakeTrainer.instances = []
        self.example = ExampleClass(
            doc_id="doc-1",
            medical_record_text="diagnosis: hypertension and diabetes",
            aci_doc_id="aci-1",
            codes=["I10"],
            diagnoses=["Hypertension"],
            starts=[11],
            ends=[23],
            evidences=["hypertension"],
        )
        self.state = {
            "doc_id": self.example.doc_id,
            "raw_input": self.example.medical_record_text,
            "is_pdf": False,
            "warnings": [],
            "trace": [],
        }

    def test_build_target(self):
        expected = json.dumps(
            {
                "codes": [
                    {
                        "diagnosis": "Hypertension",
                        "icd10_code": "I10",
                        "reasoning": "hypertension",
                        "llm_self_reported_confidence": "high",
                    }
                ]
            }
        )

        self.assertEqual(fine_tune_module._build_target(self.example), expected)

    def test_build_messages(self):
        messages = fine_tune_module._build_messages(
            self.example,
            self.state,
            build_prompt_fn=lambda _state: "PROMPT",
        )

        self.assertEqual(messages["messages"][0]["role"], "system")
        self.assertEqual(messages["messages"][1]["role"], "user")
        self.assertEqual(messages["messages"][1]["content"], "PROMPT")
        self.assertEqual(messages["messages"][2]["role"], "assistant")
        self.assertEqual(messages["messages"][2]["content"], fine_tune_module._build_target(self.example))

    def test_fine_tune_can_overfit_tiny_dataset_smoke_test(self):
        tiny_tokenizer = object()
        tiny_model = object()
        tiny_lora_config = object()
        tiny_training_args = object()

        trainer = fine_tune_module.fine_tune(
            [self.example],
            [self.example],
            pipeline_fn=lambda *_args, **_kwargs: self.state,
            build_prompt_fn=lambda _state: "PROMPT",
            tokenizer=tiny_tokenizer,
            model=tiny_model,
            lora_config=tiny_lora_config,
            training_args=tiny_training_args,
            trainer_cls=FakeTrainer,
            output_dir="/tmp/sft-overfit-smoke-test",
        )

        self.assertIsInstance(trainer, FakeTrainer)
        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_model_dir, "/tmp/sft-overfit-smoke-test")
        self.assertEqual(len(trainer.kwargs["train_dataset"]), 1)
        self.assertEqual(len(trainer.kwargs["eval_dataset"]), 1)
        self.assertEqual(trainer.kwargs["train_dataset"][0]["messages"][-1]["content"], fine_tune_module._build_target(self.example))
        self.assertEqual(trainer.memorized_targets, [fine_tune_module._build_target(self.example)])


if __name__ == "__main__":
    unittest.main()