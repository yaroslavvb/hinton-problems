# MNIST-2x5 subclass distillation

**Source:** Müller, Kornblith & Hinton (2020), "Subclass distillation".
**Demonstrates:** A teacher trained with binary super-class labels can be coaxed (via an auxiliary distance loss) to encode latent subclass structure in its logits, which a student can then use.

Re-label MNIST 0-4 as class A, 5-9 as class B. Teacher emits 10 logits (5 sub-logits per super-class) trained with binary labels + auxiliary loss maximizing pairwise distance between within-class subclass logits. Student learns the subclass structure without seeing the original 10-way labels.
