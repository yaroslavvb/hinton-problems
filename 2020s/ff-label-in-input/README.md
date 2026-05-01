# Forward-Forward: label in the first 10 pixels

**Source:** Hinton (2022).
**Demonstrates:** Supervised FF using the natural black border for label encoding. Replace the first 10 pixels with a one-hot label. Positive = (image, true label), Negative = (image, wrong label).

Reported: 4 x 2000 ReLU -> 1.36% test error after 60 epochs; with 2-pixel jittered MNIST data augmentation (25 shifts/image): 0.64% test error after 500 epochs (matches a CNN).
