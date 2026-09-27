# MNIST Nearest-Neighbor Classifier

A k-nearest-neighbor classifier for handwritten digits, written from scratch in NumPy, plus a browser demo where you draw a digit and see which training images it matched.

**95.4% test accuracy** with plain 1-NN on a balanced 7,500 / 1,000 MNIST subset. The whole test set is classified in about 0.25 s with a single vectorized distance computation.

## Try it

Draw a digit in [`docs/index.html`](docs/index.html). The page runs the same 1-NN search in JavaScript against all 7,500 training images and shows:

- the predicted label
- your drawing after MNIST-style preprocessing (fit into a 20×20 box, centered by mass in 28×28)
- the five closest training images

To run it locally, open `docs/index.html` in a browser. No server or install is needed.

## Notebook

[`mnist_nearest_neighbor.ipynb`](mnist_nearest_neighbor.ipynb) covers:

- vectorized squared-Euclidean distances using the identity ‖x−y‖² = ‖x‖² − 2x·y + ‖y‖²
- a timing comparison against scikit-learn's `KDTree` and `BallTree`
- test accuracy for k = 1…10
- the confusion matrix and misclassified examples next to the training images they matched
- per-class mean images
- exporting the training set to `docs/train.js` for the demo

```bash
pip install -r requirements.txt
jupyter notebook mnist_nearest_neighbor.ipynb
```

## Data

`MNIST/` holds a subset of the [MNIST database](http://yann.lecun.com/exdb/mnist/) (LeCun, Cortes & Burges): 750 training and 100 test images per digit, stored as flattened 784-pixel `float32` arrays with values from 0 to 255.
