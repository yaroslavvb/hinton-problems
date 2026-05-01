# Forward-Forward: top-down recurrent on repeated-frame MNIST

**Source:** Hinton (2022).
**Demonstrates:** Treat a static MNIST digit as a "video" of repeated identical frames. Each layer at time t is computed from normalized activities of layer above and below at t-1. Top layer = one-of-N label. 8 synchronous iterations with damping (0.3 old + 0.7 new). Test by running 8 iterations with each candidate label and accumulating goodness over iterations 3-5.

Reported: 1.31% test error.
