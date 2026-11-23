# digits-classifier

Compares logistic regression, an RBF SVM and a small MLP on scikit-learn's bundled 8x8 handwritten digits (1,797 images, no download needed), and lists the most confused digit pairs.

## Run
```bash
pip install -r requirements.txt
python -m pytest -q   # 4 tests
python run.py
```

## Measured result (23 Nov 2026, stratified 75/25 split, seed 0, 450 test images)
| model | accuracy |
|---|---|
| logistic regression | 0.9689 |
| RBF SVM (C=10, gamma=0.02) | 0.9889 |
| MLP (64 hidden units) | 0.9733 |

Most common SVM mistake: true 4 predicted as 8 (2 images).

## Limits
Small low-resolution images, not a deep convolutional network, and not a real-world image task. Hyperparameters were set by hand on this split, not by cross-validation, so the SVM number is somewhat optimistic. A CNN on MNIST or a medical-image set would be the next step.

## Provenance
Built with AI assistance as a learning/portfolio project.
