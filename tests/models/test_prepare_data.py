# python -m tests...
# python -m unittest discover -s tests

import unittest
from src.preprocess.load_medcoder import build_dataset, ExampleClass
from src.utils.pipeline import PipelineState
from src.models.prepare_data import _build_messages, _build_target, _get_code_description
from src.agents.coding_agent import _SYSTEM_PROMPT, _build_prompt
from src.utils.state import PipelineState
from src.utils.pipeline import run_pipeline

import requests
import json

TEXT_FILEPATH = 'data/raw_data/text.csv'
DIAGNOSIS_FILEPATH = 'data/raw_data/diagnosis.csv'
TEXT_HOLDOUT_FILEPATH = 'data/raw_data/text_holdout.csv'
DIAGNOSIS_HOLDOUT_FILEPATH = 'data/raw_data/diagnosis_holdout.csv'
OUT_DIR = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/fine_tune_data'
ICD10_API_URL = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search'

dataset = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH, 5, 10)

class TestPrepareData(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_code_description_api(self):
        params = {
            'sf': 'code,name',
            'terms': 'F32.9'
        }
        try:
            response = requests.get(ICD10_API_URL, params=params)
            data = response.json()
            num_matches = data[0]
            codes = data[1]
            results = data[3]
            # print(f'\n{num_matches}\n{codes}\n{results}')
        except Exception as e:
            print(f'An error occured: {e}')

        self.assertEqual(num_matches, 1)
        self.assertEqual(codes[0], 'F32.9')
        self.assertEqual(results[0], ['F32.9', 'Major depressive disorder, single episode, unspecified'])

    def test_get_code_description_fn(self):

        res = _get_code_description('F32.9')
        self.assertEqual(res['num_matches'], 1)
        self.assertEqual(res['code'], 'F32.9')
        self.assertEqual(res['description'], 'Major depressive disorder, single episode, unspecified')
        pass

    def test_write_json(self):
        pass

    def test_prepare(self):
        pass

dataset = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH, 6, 20)


class TestFineBuildMessages(unittest.TestCase):
    def setUp(self):
        # For example 0
        self.example: ExampleClass = dataset[0]

        # # self.state_0: PipelineState = run_pipeline(None, self.example_0.medical_record_text, False)
        # self.state: PipelineState = {
        #     "doc_id": self.example.doc_id,
        #     "raw_input": self.example.medical_record_text,
        #     "is_pdf": False,
        #     "sections": {'sections': {'chief_complaint': 'Upper respiratory infection.', 'history_of_present_illness': 'Andrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nThe patient reports that he has been doing a bit of work out in the yard in the last week or so. He started to feel really tired and short of breath. The patient denies coughing up anything, but he feels like he will soon be coughing up phlegm. He denies having a fever, but he felt a little warm. He attributed this to exerting himself. He reports that his elbows hurt quite a bit. He notes his knees were pretty tired and he felt some tension around his knees. The patient attributes these symptoms to lifting heavy bags. He has not been wearing a mask as much recently. He believes that he caught his first cold and his symptoms have worsened. He has had both of his COVID vaccinations.\n\nHe denies any history of seasonal allergies.\n\nRegarding his depression, he states it has been a crazy year and a half. He was a little concerned about that, but for the most part, he has been doing well with it. His wife got him into barre classes and this has been relaxing.\n\nRegarding his diabetes, he has been monitoring his sugar levels while he is at work, but is not as consistent on Saturdays and Sundays. His diet has been pretty good for the most part, except for some house parties. They have not been elevated since his respiratory symptoms began.\n\nThe patient has been monitoring his blood pressure at home. He reports that he is very regular with monitoring his blood pressure during the week, though less consistently on weekends. He reports his blood pressure has been under control. He has continued to utilize lisinopril 20 mg, daily.\n\nThe patient denies nausea, vomiting, diarrhea.', 'review_of_systems': '• Constitutional: Denies fever.\n\n• Cardiovascular: Endorses dyspnea on exertion.\n\n• Respiratory: Endorses shortness of breath and cough.\n\n• Gastrointestinal: Denies nausea or diarrhea.\n\n• Musculoskeletal: Endorses bilateral elbow and knee pain.\n\n• Psychiatric: Endorses depression.', 'physical_examination': '• Respiratory: Scattered rhonchi bilaterally, clears with cough.\n\n• Cardiovascular: No murmurs, gallops, or rubs. No extra heart sounds.\n\n• Musculoskeletal: Edema in the bilateral lower extremities. Pain to palpation of the bilateral elbows.', 'results': 'X-ray of the chest is unremarkable. No airspace disease. No signs of pneumonia.\n\nHemoglobin A1c is elevated at 8.', 'assessment_and_plan': 'Andrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nUpper respiratory infection.\n\n• Medical Reasoning: I believe he has contracted a viral syndrome. His chest x-ray was unremarkable and he has received both doses of the COVID-19 vaccination.\n\n• Additional Testing: We will obtain a COVID-19 test to rule this out.\n\n• Medical Treatment: I recommend he use Robitussin for cough, as well as ibuprofen or Tylenol if he develops a fever.\n\nDepression.\n\n• Medical Reasoning: He has been practicing barre classes and is doing well overall.\n\n• Medical Treatment: I offered medication or psychotherapy, but the patient opted to defer at this time.\n\nDiabetes type 2.\n\n• Medical Reasoning: His blood glucose levels have been well controlled based on home monitoring, but his recent hemoglobin A1c was elevated.\n\n• Additional Testing: We will repeat a hemoglobin A1c in 4 months.\n\n• Medical Treatment: We will increase his metformin to 1000 mg twice daily.\n\nHypertension.\n\n• Medical Reasoning: He has been compliant with lisinopril and his blood pressures have been well controlled based on home monitoring.\n\n• Additional Testing: We will order a lipid panel.\n\n• Medical Treatment: He will continue on lisinopril 20 mg once daily. This was refilled today.\n\nFollow up: I would like to see him back in approximately 4 months.\n\nPatient Agreements: The patient understands and agrees with the recommended medical treatment plan.'}, 'trace': ["[SectionAgent] detected 6 section(s): ['chief_complaint', 'history_of_present_illness', 'review_of_systems', 'physical_examination', 'results', 'assessment_and_plan']"]},
        #     "entities": {'PROBLEM': [{'text': 'depression', 'label': 'PROBLEM', 'negated': False, 'is_family': False, 'is_historical': True}, {'text': 'diabetes', 'label': 'PROBLEM', 'negated': False, 'is_family': False, 'is_historical': True}, {'text': 'hypertension', 'label': 'PROBLEM', 'negated': False, 'is_family': False, 'is_historical': True}, {'text': 'pneumonia', 'label': 'PROBLEM', 'negated': True, 'is_family': False, 'is_historical': False}], 'MEDICATION': [{'text': 'lisinopril', 'label': 'MEDICATION', 'negated': False, 'is_family': False, 'is_historical': False}, {'text': 'metformin', 'label': 'MEDICATION', 'negated': False, 'is_family': False, 'is_historical': False}], 'PROCEDURE': []},
        #     "warnings": [],
        #     "trace": [],
        # }

        self.state: PipelineState = PipelineState(
            doc_id=self.example.doc_id,
            raw_input=self.example.medical_record_text,
            is_doc=False,
            sections={'chief_complaint': 'Upper respiratory infection.', 'history_of_present_illness': 'Andrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nThe patient reports that he has been doing a bit of work out in the yard in the last week or so. He started to feel really tired and short of breath. The patient denies coughing up anything, but he feels like he will soon be coughing up phlegm. He denies having a fever, but he felt a little warm. He attributed this to exerting himself. He reports that his elbows hurt quite a bit. He notes his knees were pretty tired and he felt some tension around his knees. The patient attributes these symptoms to lifting heavy bags. He has not been wearing a mask as much recently. He believes that he caught his first cold and his symptoms have worsened. He has had both of his COVID vaccinations.\n\nHe denies any history of seasonal allergies.\n\nRegarding his depression, he states it has been a crazy year and a half. He was a little concerned about that, but for the most part, he has been doing well with it. His wife got him into barre classes and this has been relaxing.\n\nRegarding his diabetes, he has been monitoring his sugar levels while he is at work, but is not as consistent on Saturdays and Sundays. His diet has been pretty good for the most part, except for some house parties. They have not been elevated since his respiratory symptoms began.\n\nThe patient has been monitoring his blood pressure at home. He reports that he is very regular with monitoring his blood pressure during the week, though less consistently on weekends. He reports his blood pressure has been under control. He has continued to utilize lisinopril 20 mg, daily.\n\nThe patient denies nausea, vomiting, diarrhea.', 'review_of_systems': '• Constitutional: Denies fever.\n\n• Cardiovascular: Endorses dyspnea on exertion.\n\n• Respiratory: Endorses shortness of breath and cough.\n\n• Gastrointestinal: Denies nausea or diarrhea.\n\n• Musculoskeletal: Endorses bilateral elbow and knee pain.\n\n• Psychiatric: Endorses depression.', 'physical_examination': '• Respiratory: Scattered rhonchi bilaterally, clears with cough.\n\n• Cardiovascular: No murmurs, gallops, or rubs. No extra heart sounds.\n\n• Musculoskeletal: Edema in the bilateral lower extremities. Pain to palpation of the bilateral elbows.', 'results': 'X-ray of the chest is unremarkable. No airspace disease. No signs of pneumonia.\n\nHemoglobin A1c is elevated at 8.', 'assessment_and_plan': 'Andrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nUpper respiratory infection.\n\n• Medical Reasoning: I believe he has contracted a viral syndrome. His chest x-ray was unremarkable and he has received both doses of the COVID-19 vaccination.\n\n• Additional Testing: We will obtain a COVID-19 test to rule this out.\n\n• Medical Treatment: I recommend he use Robitussin for cough, as well as ibuprofen or Tylenol if he develops a fever.\n\nDepression.\n\n• Medical Reasoning: He has been practicing barre classes and is doing well overall.\n\n• Medical Treatment: I offered medication or psychotherapy, but the patient opted to defer at this time.\n\nDiabetes type 2.\n\n• Medical Reasoning: His blood glucose levels have been well controlled based on home monitoring, but his recent hemoglobin A1c was elevated.\n\n• Additional Testing: We will repeat a hemoglobin A1c in 4 months.\n\n• Medical Treatment: We will increase his metformin to 1000 mg twice daily.\n\nHypertension.\n\n• Medical Reasoning: He has been compliant with lisinopril and his blood pressures have been well controlled based on home monitoring.\n\n• Additional Testing: We will order a lipid panel.\n\n• Medical Treatment: He will continue on lisinopril 20 mg once daily. This was refilled today.\n\nFollow up: I would like to see him back in approximately 4 months.\n\nPatient Agreements: The patient understands and agrees with the recommended medical treatment plan.'},
            entities={'PROBLEM': [{'text': 'depression', 'label': 'PROBLEM', 'negated': False, 'is_family': False, 'is_historical': True}, {'text': 'diabetes', 'label': 'PROBLEM', 'negated': False, 'is_family': False, 'is_historical': True}, {'text': 'hypertension', 'label': 'PROBLEM', 'negated': False, 'is_family': False, 'is_historical': True}, {'text': 'pneumonia', 'label': 'PROBLEM', 'negated': True, 'is_family': False, 'is_historical': False}], 'MEDICATION': [{'text': 'lisinopril', 'label': 'MEDICATION', 'negated': False, 'is_family': False, 'is_historical': False}, {'text': 'metformin', 'label': 'MEDICATION', 'negated': False, 'is_family': False, 'is_historical': False}], 'PROCEDURE': []},
            codes=[
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
        )

        self.prompt: str = _build_prompt(self.state)
        self.messages: dict[str, list] = _build_messages(self.example, self.state)
        self.target = _build_target(self.example)
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
                    'icd10_description': 'Acute upper respiratory infection, unspecified',
                    'reasoning': 'Upper respiratory infection', # TODO: Find exact reasoning text, Does \n char effect reasoning text?
                    'llm_self_reported_confidence': 'high'
                },
                {
                    'diagnosis': 'Depression',
                    'icd10_code': 'F32.A',
                    'icd10_description': 'Depression, unspecified',
                    'reasoning': 'Depression',
                    'llm_self_reported_confidence': 'high'
                },
                {
                    'diagnosis': 'Diabetes type 2',
                    'icd10_code': 'E11.9',
                    'icd10_description': 'Type 2 diabetes mellitus without complications',
                    'reasoning': 'Diabetes type 2',
                    'llm_self_reported_confidence': 'high'
                },
                {
                    'diagnosis': 'Hypertension',
                    'icd10_code': 'I10',
                    'icd10_description': 'Essential (primary) hypertension',
                    'reasoning': 'Hypertension',
                    'llm_self_reported_confidence': 'high'
                }
            ]
        })

        self.assertEqual(self.target, true_target_0)
        pass

    def test_build_messages(self):
        '''Test that _build_messages() creates the correct message format for '''
        # print(self.prompt_0)
        true_messages = {'messages': [{'role': 'system', 'content': 'You are a clinical coding assistant supporting a human medical coder.\nYou will be given clinical entities already extracted from a note (problems,\nmedications, procedures), each marked as negated or not. You must NOT invent\nfindings that are not in the provided entity list or section text.\n\nFor each ACTIVE (non-negated) problem, suggest one plausible ICD-10-CM code.\nIf you are not confident of the exact code, give your best general-category\ncode and say so in the reasoning.\n\nICD-10-CM codes must be formatted exactly as standard ICD-10-CM codes:\n- Use uppercase letters and digits only, with no spaces or extra punctuation.\n- The code must start with 1 letter followed by 2 digits.\n- If the code has more than 3 characters, place a decimal point after the first 3 characters.\n- Do not add leading zeros, trailing zeros, or any unlisted characters.\n- Valid examples: I10, E11.9, J44.9, Z79.4\n- Invalid examples: e119, E11 9, E11-9, 11.9, E11.90\n\nReturn ONLY valid JSON, no markdown fences, no preamble, matching this shape:\n{\n  "codes": [\n    {\n\t\t"diagnosis": "string, the clinical concept",\n\t\t"icd10_code": "string",\n\t\t"icd10_description": "string",\n\t\t"reasoning": "string, 1-2 sentences, must reference only the given entities/sections",\n\t\t"llm_self_reported_confidence": "high" | "medium" | "low"\n    }\n  ]\n}\n'}, {'role': 'user', 'content': '\n\t\tACTIVE PROBLEMS (extracted, not negated):\n\t\t[\n  "depression",\n  "diabetes",\n  "hypertension"\n]\n\n\t\tEXPLICITLY NEGATED / RULED-OUT (do NOT code these):\n\t\t[\n  "pneumonia"\n]\n\n\t\tMEDICATIONS ON RECORD:\n\t\t[\n  "lisinopril",\n  "metformin"\n]\n\n\t\tPROCEDURES ON RECORD:\n\t\t[]\n\n\t\tRELEVANT NOTE TEXT (Assessment/Plan or main body):\n\t\tAndrew Campbell is a 59-year-old male with a past medical history significant for depression, type 2 diabetes, and hypertension. He presents today with an upper respiratory infection.\n\nUpper respiratory infection.\n\n• Medical Reasoning: I believe he has contracted a viral syndrome. His chest x-ray was unremarkable and he has received both doses of the COVID-19 vaccination.\n\n• Additional Testing: We will obtain a COVID-19 test to rule this out.\n\n• Medical Treatment: I recommend he use Robitussin for cough, as well as ibuprofen or Tylenol if he develops a fever.\n\nDepression.\n\n• Medical Reasoning: He has been practicing barre classes and is doing well overall.\n\n• Medical Treatment: I offered medication or psychotherapy, but the patient opted to defer at this time.\n\nDiabetes type 2.\n\n• Medical Reasoning: His blood glucose levels have been well controlled based on home monitoring, but his recent hemoglobin A1c was elevated.\n\n• Additional Testing: We will repeat a hemoglobin A1c in 4 months.\n\n• Medical Treatment: We will increase his metformin to 1000 mg twice daily.\n\nHypertension.\n\n• Medical Reasoning: He has been compliant with lisinopril and his blood pressures have been well controlled based on home monitoring.\n\n• Additional Testing: We will order a lipid panel.\n\n• Medical Treatment: He will continue on lisinopril 20 mg once daily. This was refilled today.\n\nFollow up: I would like to see him back in approximately 4 months.\n\nPatient Agreements: The patient understands and agrees with the recommended medical treatment plan.\n\t'}, {'role': 'assistant', 'content': '{"codes": [{"diagnosis": "Upper respiratory infection", "icd10_code": "J06.9", "icd10_description": "Acute upper respiratory infection, unspecified", "reasoning": "Upper respiratory infection", "llm_self_reported_confidence": "high"}, {"diagnosis": "Depression", "icd10_code": "F32.A", "icd10_description": "Depression, unspecified", "reasoning": "Depression", "llm_self_reported_confidence": "high"}, {"diagnosis": "Diabetes type 2", "icd10_code": "E11.9", "icd10_description": "Type 2 diabetes mellitus without complications", "reasoning": "Diabetes type 2", "llm_self_reported_confidence": "high"}, {"diagnosis": "Hypertension", "icd10_code": "I10", "icd10_description": "Essential (primary) hypertension", "reasoning": "Hypertension", "llm_self_reported_confidence": "high"}]}'}]} 
        print('\n\n=======\n\n', str(self.messages), '\n\n\n======\n\n', str(true_messages))
        self.assertEqual(
            self.messages, true_messages
        )
        pass


if __name__ == '__main__':
    unittest.main()