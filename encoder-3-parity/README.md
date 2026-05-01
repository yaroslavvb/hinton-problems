# 3-bit even-parity ensemble

**Source:** Ackley, Hinton & Sejnowski (1985), "A learning algorithm for Boltzmann machines", Cognitive Science.
**Demonstrates:** Why hidden units are necessary — pairwise statistics carry zero information about parity.

Visible-only Boltzmann machine on the 4-pattern even-parity ensemble (each pattern p=0.25). The training distribution has zero pairwise correlation between any two visible units; learning fails without hidden units. Motivates the encoder problems that follow.
