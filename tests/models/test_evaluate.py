# python -m tests...
# python -m unittest discover -s tests

import unittest
from src.preprocess.load_medcoder import build_dataset, ExampleClass
from src.utils.pipeline import PipelineState
from src.models.evaluate import multilabel_metrics

TEXT_FILEPATH = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/raw_data/text.csv'
DIAGNOSIS_FILEPATH = '/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/raw_data/diagnosis.csv'

dataset = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH, None, None, 5, 10, None, None)

class TestBuildDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_multilabel_metrics_toy_set(self):
        true_labels = [
            ['cat', 'dog'],
            ['cat', 'bird'],
            ['bird'],
            []
        ]
        pred_labels = [
            ['dog', 'bird'],
            ['cat'],
            ['bird'],
            ['cat', 'bird']
        ]
        metrics = multilabel_metrics(true_labels, pred_labels)

        self.assertAlmostEqual(metrics['micro_precision'], 0.500)
        self.assertAlmostEqual(metrics['micro_recall'], 0.600)
        self.assertAlmostEqual(metrics['micro_f1'], 0.545)
        self.assertAlmostEqual(metrics['macro_precision'], 0.611)
        self.assertAlmostEqual(metrics['macro_recall'], 0.667)
        self.assertAlmostEqual(metrics['macro_f1'], 0.633)
        self.assertEqual(metrics['num_notes'], 4)
        self.assertEqual(metrics['num_unique_codes'], 3)
        pass

    def test_multilabel_icd10_codes(self):
        true_labels = [
            ['J06.9', 'F32.A', 'E11.9', 'I10'],
            ['M06.9', 'I48.91', 'K21.9', 'F32.9'],
            ['E11.65', 'F32.A', 'Z94.0']
        ]
        pred_labels = [
            ['F32.9', 'E11.9', 'I10'],
            ['F32.9', 'I48.91'],
            ['F32.9', 'E11.9', 'Z94.0', 'I10']
        ]
        metrics = multilabel_metrics(true_labels, pred_labels)

        self.assertAlmostEqual(metrics['micro_precision'], 0.556)
        self.assertAlmostEqual(metrics['micro_recall'], 0.455)
        self.assertAlmostEqual(metrics['micro_f1'], 0.500)
        self.assertAlmostEqual(metrics['macro_precision'], 0.333)
        self.assertAlmostEqual(metrics['macro_recall'], 0.500)
        self.assertAlmostEqual(metrics['macro_f1'], 0.383)
        self.assertEqual(metrics['num_notes'], 3)
        self.assertEqual(metrics['num_unique_codes'], 10)

        pass


if __name__ == '__main__':
    unittest.main()