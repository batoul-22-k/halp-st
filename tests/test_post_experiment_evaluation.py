"""Statistical and scoring invariants for the post-experiment pipeline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from post_experiment_evaluation import paired_statistics, grading_metrics
from evaluation_common import METRICS, answers
from src.evaluation.application_scoring import application_score

def test_paired_statistics_known_differences():
    x = np.array([1., 2., 3., 4., 5.])
    result = paired_statistics(pd.DataFrame({m: x for m in METRICS})).iloc[0]
    assert result.mean_paired_difference == 3
    assert result.std_paired_difference == pytest.approx(np.sqrt(2.5))
    assert result.t_pvalue == pytest.approx(stats.ttest_rel(x, np.zeros(5)).pvalue)
    assert result.wilcoxon_pvalue == .0625
    assert result.cohens_dz == pytest.approx(3/np.sqrt(2.5))
    assert result.ci95_low < 3 < result.ci95_high
    assert result.t_pvalue_holm == pytest.approx(min(1, result.t_pvalue*5))

def test_macro_metrics_keep_absent_classes():
    result = grading_metrics(['correct'], ['correct'])
    assert result['accuracy'] == 1
    assert result['macro_f1'] == pytest.approx(1/3)
    assert result['partial_support'] == 0

def test_labels_are_authoritative_and_invalid_data_rejected(tmp_path):
    path = tmp_path/'answers.csv'
    path.write_text('reference_answer,student_answer,label,score\na,b,partial,0.25\n')
    assert answers(path).true_label.tolist() == ['partial']
    path.write_text('expected_answer,student_answer,true_label\na,b,unknown\n')
    with pytest.raises(ValueError):
        answers(path)

def test_fixed_thresholds_and_lexical_overlap():
    assert application_score(1, 'a', 'b')['label'] == 'correct'
    assert application_score(.5/.7, 'a', 'b')['label'] == 'partial'
    assert application_score(.49/.7, 'a', 'b')['label'] == 'incorrect'
    assert application_score(0, 'a b', 'a c')['lexical_score'] == pytest.approx(1/3)
