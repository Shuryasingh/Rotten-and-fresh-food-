# Evaluation Reports

This directory contains detailed evaluation outputs and metrics from `evaluate.py` and `train.py --evaluate`.

## Files

- `evaluation_val.json`: Evaluation results on the validation split.
- `evaluation_test.json`: Evaluation results on the test split.

Each evaluation file contains:
- Split name & image counts
- Confidence threshold used
- Confusion matrix (2x2 forced-choice & thresholded)
- Overall forced-choice accuracy
- Rotten recall (Critical metric)
- Fresh recall & precision
- Accepted vs uncertain fraction
- Accuracy on accepted predictions
- List of individual misclassifications with confidence scores
