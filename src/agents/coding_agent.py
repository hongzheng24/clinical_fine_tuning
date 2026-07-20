'''
Agent 4: Coding Recommendation
'''

import json
from src.models.load_models.base_model import generate_json
from src.utils.state import PipelineState


# Edited prompt
_SYSTEM_PROMPT = '''You are a clinical coding assistant supporting a human medical coder.
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
}
'''

# # Original prompt
# _SYSTEM_PROMPT = '''You are a clinical coding assistant supporting a human medical coder.
# You will be given clinical entities already extracted from a note (problems,
# medications, procedures), each marked as negated or not. You must NOT invent
# findings that are not in the provided entity list or section text.

# For each ACTIVE (non-negated) problem, suggest one plausible ICD-10-CM code.
# If you are not confident of the exact code, give your best general-category
# code and say so in the reasoning.

# Return ONLY valid JSON, no markdown fences, no preamble, matching this shape:
# {
#   "codes": [
#     {
# 		"diagnosis": "string, the clinical concept",
# 		"icd10_code": "string",
# 		"icd10_description": "string",
# 		"reasoning": "string, 1-2 sentences, must reference only the given entities/sections",
# 		"llm_self_reported_confidence": "high" | "medium" | "low"
#     }
#   ]
# }
# '''

def _build_prompt(state: PipelineState) -> str:
	'''
    Build LLM prompt specifiying active problems, negated problems, medications, procedures, and relevant notes.
	Instruct model to return diagnoses, ICD10 codes, ICD10 descriptions, reasoning, and self-report confidence.

	Parameters
	==========
	state: PipelineState
		State containing entity from which to build a prompt.
	
	Returns
	=======
	prompt: str
		Completed LLM prompt.
	'''
	entities = state.get('entities', {})
	sections = state.get('sections', {})

	active_problems = [entity["text"] for entity in entities.get("PROBLEM", []) if not entity["negated"]]
	negated_problems = [entity["text"] for entity in entities.get("PROBLEM", []) if entity["negated"]]
	medications = [entity["text"] for entity in entities.get("MEDICATION", [])]
	procedures = [entity["text"] for entity in entities.get("PROCEDURE", [])]
	assessment_plan = sections.get("assessment_and_plan") or sections.get("PREAMBLE", "")

	return f'''
		ACTIVE PROBLEMS (extracted, not negated):
		{json.dumps(active_problems, indent=2)}

		EXPLICITLY NEGATED / RULED-OUT (do NOT code these):
		{json.dumps(negated_problems, indent=2)}

		MEDICATIONS ON RECORD:
		{json.dumps(medications, indent=2)}

		PROCEDURES ON RECORD:
		{json.dumps(procedures, indent=2)}

		RELEVANT NOTE TEXT (Assessment/Plan or main body):
		{assessment_plan[:3000]}
	'''

def recommend_codes(state: PipelineState) -> dict:
	'''
	Ask LLM to generate diagnoses, ICD10 codes, ICD10 descriptions, reasoning, and self-report confidence
	given a patient's clinical records.

	Parameters
	==========
	state:
		State containing entity for which to generate codes.
	
	Returns
	=======
	codes_dict: dict
		Map containing ICD10 codes.
	'''
	try:
		response = generate_json(_SYSTEM_PROMPT, _build_prompt(state))
		codes = response.get('codes', [])
		return {"codes": codes, "trace": [f"[CodingAgent] LLM suggested {len(codes)} ICD-10 code(s)"]}
	except Exception as exception:
		return {
			"codes": [],
			"warnings": [f"CodingAgent: LLM call/parse failed ({exception}) -- no codes generated"],
			"trace": [f"[CodingAgent] failed: {exception}"],
		}
