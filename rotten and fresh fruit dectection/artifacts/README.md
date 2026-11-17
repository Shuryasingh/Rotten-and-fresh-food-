# Model Artifacts

This directory stores outputs produced by `train.py` and `export_model.py`.

## Expected Files

- `fruit.onnx`: Exported ONNX model used for fast OpenCV DNN inference on laptop and Raspberry Pi.
- `metadata.json`: Model configuration, class order (`["fresh", "rotten"]`), input size (`224`), training timestamp, default threshold, and model architecture.
- `best_model.pt`: Best PyTorch model checkpoint saved based on validation loss/accuracy.
- `training_history.json`: Epoch-by-epoch loss and accuracy history.
- `training_graph.png`: Loss and accuracy curves over training epochs.
- `validation_report.json`: Detailed validation metrics including confusion matrix and rotten recall.

> [!NOTE]
> In an initial clean checkout, this folder contains no trained model.
> You must collect or provide dataset images and run `python train.py` to generate these artifacts.
