import pandas as pd
from __future__ import annotations

import re
from dataclasses import dataclass, field
from src.utils.pipeline import run_pipeline
from src.utils.state import PipelineState

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from src.preprocess.load_asclepius import ExampleClass



def multilabel_metrics(true_codes: list[list[str]], pred_codes: list[list[str]]) -> dict[str, int | float]:
    '''
    Calculate micro/macro precision, recall, F1 score given two sets of true codes and predicted codes.
    Use sklearn.preprocessing.MultiLabelBinarizer to create a sparse binary matrix mapping notes to codes.

    Parameters
    ==========
    true_codes: list[list[str]]
        List of true codes for each clinical note.
    pred_codes: list[list[str]]
        List of predicted codes for each clinical note.

    Returns
    =======
    metrics: dict[str, int | float]
        Dict containing micro/macro precisions, recall, and F1 scores.
    '''
    assert len(true_codes) == len(pred_codes)

    mlb = MultiLabelBinarizer().fit(true_codes + pred_codes)
    y_true, y_pred = mlb.fit_transform(true_codes), mlb.fit_transform(pred_codes)
    
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    return {
        "micro_precision": round(micro_p, 3), "micro_recall": round(micro_r, 3), "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_p, 3), "macro_recall": round(macro_r, 3), "macro_f1": round(macro_f1, 3),
        "num_notes": len(true_codes), "num_unique_codes": len(mlb.classes_),
    }


def _category(code: str) -> str:
    '''
    Get the category of an ICD-10 code (first 3 characters). 
    Example: 'E11.65' -> 'E11'.

    Parameters
    ==========
    code: str
        ICD 10 code.
    
    Returns
    =======
    category: str
        Category of code.
    '''
    return re.split(r"\.", code, maxsplit=1)[0][:3]


def code_metrics_report(true_codes: list[list[str]], pred_codes: list[list[str]]) -> dict[str, float | dict[str, int | float]]:
    '''
    Calculate two metrics: one for exact codes and one for code categories only.
    Determine the specificity F1 score gap which is the difference between the two F1 scores.

    Parameters
    ==========
    true_codes: list[list[str]]
        List of true codes for each clinical note.
    pred_codes: list[list[str]]
        List of predicted codes for each clinical note.

    Returns
    =======
    report: dict[str, float | dict[str, int | float]]
        Dict containing exact-code-level metrics, category-level metrics, and specificity F1 gap.
    '''
    exact = multilabel_metrics(true_codes, pred_codes)
    category = multilabel_metrics(
        [[_category(code) for code in codes] for codes in true_codes],
        [[_category(code) for code in codes] for codes in pred_codes],
    )
    return {
        "exact_code_level": exact,
        "category_level": category,
        "specificity_gap_f1": round(category["micro_f1"] - exact["micro_f1"], 3),
    }


def evaluate(states: list[PipelineState]):
    '''
    Parameters
    ==========
    states: list[PipelineStates]
        List of final states produced by pipeline to be evaluated.

    Returns
    =======
    eval_dict: dict[str, dict]
        Dict containing set metrics, ... # TODO: Additional metrics
    '''
    return {
        'set_metrics': code_metrics_report(states), # TODO: Additional metrics
    }


'''
# Overall Pseudocode
1. Run pipeline with base model and record final states
2. Run pipeline with fine-tuned model and record final states
3. Run evaluation methods on both sets of states


base_final_states, fine_tune_final_states = [], []
for example in eval_examples:
    base_final_states.append(run_pipeline(example, model='base'))
    fine_tune_final_states.append(run_pipeline(example, model='fine-tune'))

base_metrics = hierarichal_report(base_final_states)
fine_tune_metrics = hierarichal_report(fine_tune_final_states)
'''



        

        




    
