# Forward-Forward: CIFAR-10 with locally-connected layers

**Source:** Hinton (2022).
**Demonstrates:** FF closes the gap with backprop on cluttered images. Two or three hidden layers with a 32x32 topographic map, 3 channels per location, 11x11 receptive fields with bottom-up + top-down inputs. Backprop baseline 37-39%; FF variants 41-46%. The FF-backprop gap does not widen with depth.
