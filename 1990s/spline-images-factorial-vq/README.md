# Spline images (intrinsic dim 5)

**Source:** Hinton & Zemel (1994), "Autoencoders, MDL and Helmholtz free energy", NIPS 6.
**Demonstrates:** Factorial vector quantization beats standard stochastic VQ on data with multiple independent latent factors. Introduces the bits-back / Helmholtz free-energy view.

200 images (8x12) formed by Gaussian-blurring a spline through 5 control points (intrinsic dim 5). Compared standard 24-unit stochastic VQ, 4 separate stochastic VQs, factorial VQ (4 dimensions x 6 hidden units), PCA. Factorial VQ achieves 18-bit reconstruction + 7-bit code.
