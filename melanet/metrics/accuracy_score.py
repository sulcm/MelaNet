import evaluate
import datasets
import numpy as np

from sklearn.metrics._classification import _check_set_wise_labels, _check_zero_division
from sklearn.metrics import multilabel_confusion_matrix


def _safe_divide(num, denom):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            num,
            denom,
            out=np.zeros_like(num, dtype=float),
            where=denom != 0
        )


def accuracy_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=None,
    average=None,
    sample_weight=None,
    zero_division="warn",
    **kwargs
):
    """
    Compute accuracy score for binary, multiclass, and multilabel problems.

    This function generalizes accuracy computation to support different averaging
    strategies: 'micro', 'macro', 'weighted', or None.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_labels)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_labels)
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        The set of labels to include. Labels present in the data can be excluded.
        By default, all labels in y_true and y_pred are used in sorted order.

    average : {'micro', 'macro', 'weighted', None}, default=None
        Determines the type of averaging performed on the data:
        - None: return accuracy per class (for multiclass/multilabel).
        - 'micro': global accuracy across all labels.
        - 'macro': mean accuracy across labels.
        - 'weighted': mean accuracy weighted by support.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    accuracy : float or ndarray of float
        Accuracy score. If `average=None`, returns an array of accuracies per class.
        Otherwise, returns the averaged accuracy.
    """
    _check_zero_division(zero_division)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine valid labels and create confusion matrices
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label=pos_label)
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true, y_pred, labels=labels, sample_weight=sample_weight, samplewise=samplewise
    )

    # Extract counts
    tp = MCM[:, 1, 1]
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0]

    # Handle different averaging strategies
    if average in ("none", None):
        return _safe_divide(tp + tn, tp + tn + fp + fn)
    elif average == "micro":
        # Global accuracy over all instances
        return float(
            tp.sum() / y_true.shape[0]
        )
    else:
        # Mean accuracy across labels
        if average == "samples":
            weights = sample_weight
            if y_true.ndim == 1:
                _acc = y_true == y_pred
            else:
                _acc = np.mean(y_true == y_pred, axis=1)
        else:
            # Average: macro, weighted
            if average == "weighted":
                _support = tp + fn
                weights = _support / np.sum(_support) if np.sum(_support) > 0 else None
            else:
                weights = None
            # _acc = _safe_divide(tp, fp + tp)
            _acc = _safe_divide(tp + tn, fp + tp + tn + fn)
        return float(
            np.average(_acc, weights=weights)
        )


_DESCRIPTION = """
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    labels (`list`, default=`None`):
        The set of labels to include and their order if
        `average is None`. For multilabel classification, these are column indices.
    pos_label (`int`, default=`None`): The class to be considered the positive class,
        in the case where `average` is set to `binary`.
    average ({'micro', 'macro', 'samples', 'weighted', `None`}, default=`None`):
        Determines the type of averaging performed:
        - ``None``: return per-class accuracies.
        - ``'micro'``: compute global accuracy over all samples and labels.
        - ``'macro'``: unweighted mean of accuracies over labels.
        - ``'weighted'``: mean of accuracies weighted by support (true label counts).
        - ``'samples'``: mean accuracy over samples (useful for multilabel).
    sample_weight (`list` of `float`, default=`None`): array-like sample weights of shape (n_samples,).
    zero_division ({"warn", 0.0, 1.0, np.nan}, default="warn"):
        Sets the value to return when division by zero occurs.
Returns:
    accuracy (`float` or `int`): Accuracy score. A higher score means higher accuracy.
"""

_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=None, average=None, sample_weight=None, zero_division="warn", **kwargs):
        acc_score = accuracy_score(
            references,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division
        )
        return {
            "accuracy": acc_score
        }