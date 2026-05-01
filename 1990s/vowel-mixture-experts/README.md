# Vowel discrimination (Peterson-Barney)

**Source:** Jacobs, Jordan, Nowlan & Hinton (1991), "Adaptive mixtures of local experts", Neural Computation.
**Demonstrates:** Mixture of experts cleanly partitions input space, reaching 90% test accuracy in half the epochs of monolithic backprop.

4-class speaker-independent classification of [i], [I], [a], [Lambda] from F1, F2 formants; 75 speakers (Peterson-Barney 1952). Mixture of 4 (or 8) linear experts plus softmax gating.
