# Shifter / shift-direction inference

**Source:** Hinton & Sejnowski (1986), "Learning and relearning in Boltzmann machines", PDP Vol. 1, Ch. 7.
**Demonstrates:** Boltzmann learning discovers third-order conjunctive features unattainable to perceptrons. The canonical "higher-order feature" toy.

Visible split into:
- V1: 8 input bits, p=0.3 each.
- V2: V1 shifted by -1, 0, or +1 (with wraparound).
- V3: 3 one-hot units giving the shift direction.

Pairwise statistics carry zero information about shift; 24 hidden units must therefore find conjunctive features ("two large negative weights flanked by excitatory weights"). Reported recognition accuracy: 50-89% depending on number of active V1 bits.
