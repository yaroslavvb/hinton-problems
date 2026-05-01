# Forward-Forward: hybrid-image MNIST negatives

**Source:** Hinton (2022), "The forward-forward algorithm: some preliminary investigations", arXiv / NeurIPS keynote.
**Demonstrates:** Unsupervised FF using hand-crafted "hybrid image" negatives. Negatives mix two different digits with a smoothly thresholded random mask (repeated [1/4, 1/2, 1/4] blurring of a random bit-mask, thresholded at 0.5). Mask creates large coherent regions, preserving short-range correlations but destroying long-range shape correlations — forcing FF to learn shape features rather than texture.

Reported: 1.37% test error using last 3 hidden layers' normalized activities as input to a softmax classifier (1.16% with locally-connected layers + peer normalization).
