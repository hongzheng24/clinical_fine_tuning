import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class ExampleClass:
	doc_id: str
	medical_record_text: str
	aci_doc_id: str
	codes: list[str]
	diagnoses: list[str]
	starts: list[str]
	ends: list[str]
	evidences: list[str]


def build_dataset(
		text_filepath: str,
		diagnosis_filepath: str,
		text_holdout_filepath: str=None,
		diagnosis_holdout_filepath: str=None,
		text_nrows: int=None,
		diagnosis_nrows: int=None,
		text_holdout_nrows: int=None,
		diagnosis_holdout_nrows: int=None,
	) -> list[ExampleClass]:
	'''
	Build a list of ExampleClass objects that represent each clinical case from the MedCodER dataset.

	Parameters
	==========
	text_filepath: str
		Filepath to text.csv which contains clinical text records, used as a test set.
	diagnosis_filepath: str
		Filepath to diagnosis.csv which contains ICD10 codes, diagnoses, and evidence text indices, used as a test set.
	text_holdout_filepath: str
		Filepath to text_holdout.csv which contains clinical text records, used as a holdout set for user experiments.
	diagnosis_holdout_filepath: str
		Filepath to diagnosis_holdout.csv which contains ICD10 codes, diagnoses, and evidence text indices, used as a holdout set for user experiments.
	text_nrows: int | None (default=None)
		(Optional) Number of rows of text.csv file to read.
	diagnosis_nrows: int | None (default=None)
		(Optional) Number of rows of diagnosis.csv file to read.
	text_holdout_nrows: int | None (default=None)
		(Optional) Number of rows of text_holdout.csv file to read.
	diagnosis_holdout_nrows: int | None (default=None)
		(Optional) Number of rows of diagnosis_holdout.csv file to read.

	Returns
	=======
	dataset: list[ExampleClass]
		List containing examples of patient cases, represented by an ExampleClass object containing
		these parameters: doc_id, medical_record_text, aci_doc_id, codes, diagnoses, starts, ends
	'''


	text_df = pd.read_csv(text_filepath, nrows=text_nrows)
	diagnosis_df = pd.read_csv(diagnosis_filepath, nrows=diagnosis_nrows)
	# text_holdout_df = pd.read_csv(text_holdout_filepath, nrows=text_holdout_nrows)
	# diagnosis_holdout_df = pd.read_csv(diagnosis_holdout_filepath, nrows=diganosis_handout_nrows)


	dataset = []
	for _, text_row in text_df.iterrows():

		doc_id, medical_record_text, aci_doc_id = text_row['Document ID'], text_row['medical_record_text'], text_row['aci_doc_id']

		labels = diagnosis_df.loc[diagnosis_df['Document ID'] == doc_id]
		codes, diagnoses, starts, ends, evidences = [], [], [], [], []

		for _, labels_row in labels.iterrows(): # TODO: Check df.loc()
			doc_id, code, diagnosis, start, end = labels_row

			codes.append(code)
			diagnoses.append(diagnosis)
			starts.append(start)
			ends.append(end)
			evidences.append(medical_record_text[start:end])
		
		assert(len(codes) == len(diagnoses) == len(starts) == len(ends))

		dataset.append(ExampleClass(
			doc_id,
			medical_record_text,
			aci_doc_id,
			codes,
			diagnoses,
			starts,
			ends,
			evidences
		))
	
	return dataset


"""
# Deprecated build_dataset(), returned dict.
def build_dataset(text_filepath, diagnosis_filepath) -> dict:
	'''
	Dataset structure:

	dataset: dict[str, dict[str: str | list[str]] = doc_id: str -> {
		doc_id: str,
		medical_record_text: str,
		aci_doc_id: str,
		codes: list[str],
		diagnoses: list[str]
	}

	[Deprecated]
	{doc_id -> {
		doc_id: str,
		medical_record_text: str,
		aci_doc_id: str,
		labels: dict[list, str] = {
			codes: list[str],
			diagnoses: list[str],
			start: str,
			end: str
		}
	}
	'''
	text_df = pd.read_csv(text_filepath)
	diagnosis_df = pd.read_csv(diagnosis_filepath)

	# print(text_df.iloc[0])
	# print(text_df.index)

	dataset = defaultdict(dict)
	for _, row in text_df.iterrows():
		doc_id, text, aci_doc_id = row['Document ID'], row['medical_record_text'], row['aci_doc_id']

		labels = diagnosis_df.loc[diagnosis_df['Document ID'] == doc_id]
		codes, diagnoses = [], []

		for _, row in labels.iterrows(): # TODO: Check df.loc()
			doc_id, code, diagnosis, start, end = row

			codes.append(code)
			diagnoses.append(diagnosis)

		dataset[doc_id] = {
			'doc_id': doc_id,
			'medical_record_text': text,
			'aci_doc_id': aci_doc_id,
			'codes': codes,
			'diagnoses': diagnoses
		}
	
	return dataset
"""