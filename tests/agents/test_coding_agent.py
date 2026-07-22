# python -m tests...
# python -m unittest discover -s tests

import unittest
from src.preprocess.load_medcoder import build_dataset, ExampleClass
from src.utils.pipeline import PipelineState

TEXT_FILEPATH = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/raw_data/text.csv'
DIAGNOSIS_FILEPATH = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/raw_data/diagnosis.csv'

dataset = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH, None, None, 5, 10, None, None)

class TestBuildDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test(self)


if __name__ == '__main__':
    unittest.main()