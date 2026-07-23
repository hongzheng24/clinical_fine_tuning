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

class TestBuildDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_dataset_first_example(self):
        '''
        Test the first example. Test Example attributes.
        '''

        example = dataset[0]
        self.assertTrue(
            example.doc_id == '240503191033444EHGFED'
            and example.medical_record_text == 'CHIEF COMPLAINT\n\nUpper respiratory infection.\n\nHISTORY OF PRESENT ILLNESS\n\nAndrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nThe patient reports that he has been doing a bit of work out in the yard in the last week or so. He started to feel really tired and short of breath. The patient denies coughing up anything, but he feels like he will soon be coughing up phlegm. He denies having a fever, but he felt a little warm. He attributed this to exerting himself. He reports that his elbows hurt quite a bit. He notes his knees were pretty tired and he felt some tension around his knees. The patient attributes these symptoms to lifting heavy bags. He has not been wearing a mask as much recently. He believes that he caught his first cold and his symptoms have worsened. He has had both of his COVID vaccinations.\n\nHe denies any history of seasonal allergies.\n\nRegarding his depression, he states it has been a crazy year and a half. He was a little concerned about that, but for the most part, he has been doing well with it. His wife got him into barre classes and this has been relaxing.\n\nRegarding his diabetes, he has been monitoring his sugar levels while he is at work, but is not as consistent on Saturdays and Sundays. His diet has been pretty good for the most part, except for some house parties. They have not been elevated since his respiratory symptoms began.\n\nThe patient has been monitoring his blood pressure at home. He reports that he is very regular with monitoring his blood pressure during the week, though less consistently on weekends. He reports his blood pressure has been under control. He has continued to utilize lisinopril 20 mg, daily.\n\nThe patient denies nausea, vomiting, diarrhea.\n\nREVIEW OF SYSTEMS\n\n• Constitutional: Denies fever.\n\n• Cardiovascular: Endorses dyspnea on exertion.\n\n• Respiratory: Endorses shortness of breath and cough.\n\n• Gastrointestinal: Denies nausea or diarrhea.\n\n• Musculoskeletal: Endorses bilateral elbow and knee pain.\n\n• Psychiatric: Endorses depression.\n\nPHYSICAL EXAMINATION\n\n• Respiratory: Scattered rhonchi bilaterally, clears with cough.\n\n• Cardiovascular: No murmurs, gallops, or rubs. No extra heart sounds.\n\n• Musculoskeletal: Edema in the bilateral lower extremities. Pain to palpation of the bilateral elbows.\n\nRESULTS\n\nX-ray of the chest is unremarkable. No airspace disease. No signs of pneumonia.\n\nHemoglobin A1c is elevated at 8.\n\nASSESSMENT AND PLAN\n\nAndrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nUpper respiratory infection.\n\n• Medical Reasoning: I believe he has contracted a viral syndrome. His chest x-ray was unremarkable and he has received both doses of the COVID-19 vaccination.\n\n• Additional Testing: We will obtain a COVID-19 test to rule this out.\n\n• Medical Treatment: I recommend he use Robitussin for cough, as well as ibuprofen or Tylenol if he develops a fever.\n\nDepression.\n\n• Medical Reasoning: He has been practicing barre classes and is doing well overall.\n\n• Medical Treatment: I offered medication or psychotherapy, but the patient opted to defer at this time.\n\nDiabetes type 2.\n\n• Medical Reasoning: His blood glucose levels have been well controlled based on home monitoring, but his recent hemoglobin A1c was elevated.\n\n• Additional Testing: We will repeat a hemoglobin A1c in 4 months.\n\n• Medical Treatment: We will increase his metformin to 1000 mg twice daily.\n\nHypertension.\n\n• Medical Reasoning: He has been compliant with lisinopril and his blood pressures have been well controlled based on home monitoring.\n\n• Additional Testing: We will order a lipid panel.\n\n• Medical Treatment: He will continue on lisinopril 20 mg once daily. This was refilled today.\n\nFollow up: I would like to see him back in approximately 4 months.\n\nPatient Agreements: The patient understands and agrees with the recommended medical treatment plan.'
            and example.aci_doc_id == 'test_1'
            and example.codes == ['J06.9', 'F32.A', 'E11.9', 'I10']
            and example.diagnoses == ['Upper respiratory infection', 'Depression', 'Diabetes type 2', 'Hypertension']
            and example.starts == [2749, 3131, 3336, 3642]
            and example.ends == [2776, 3141, 3351, 3654]
        )

        self.assertTrue(len(example.codes) == len(example.diagnoses) == len(example.starts) == len(example.ends))
    
class TestFineTune(unittest.TestCase):
    def setUp(self):
        # For example 0
        self.example_0: ExampleClass = dataset[0]
        self.state_0: PipelineState = run_pipeline(None, self.example_0.medical_record_text, False)
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
                {"role": "user", "content": '''                ACTIVE PROBLEMS (extracted, not negated):
                                            [
                                            "diabetes",
                                            "hypertension"
                                            ]

                                                            EXPLICITLY NEGATED / RULED-OUT (do NOT code these):
                                                            [
                                            "pneumonia"
                                            ]

                                                            MEDICATIONS ON RECORD:
                                                            [
                                            "lisinopril",
                                            "metformin"
                                            ]

                                                            PROCEDURES ON RECORD:
                                                            []

                                                            RELEVANT NOTE TEXT (Assessment/Plan or main body):
                                                            CHIEF COMPLAINT

                                            Upper respiratory infection.

                                            HISTORY OF PRESENT ILLNESS

                                            Andrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.

                                            The patient reports that he has been doing a bit of work out in the yard in the last week or so. He started to feel really tired and short of breath. The patient denies coughing up anything, but he feels like he will soon be coughing up phlegm. He denies having a fever, but he felt a little warm. He attributed this to exerting himself. He reports that his elbows hurt quite a bit. He notes his knees were pretty tired and he felt some tension around his knees. The patient attributes these symptoms to lifting heavy bags. He has not been wearing a mask as much recently. He believes that he caught his first cold and his symptoms have worsened. He has had both of his COVID vaccinations.'''
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