# Bars task

**Source:** Hinton, Dayan, Frey & Neal (1995), "The wake-sleep algorithm", Science.
**Demonstrates:** Foundational factorial-causal toy for the wake-sleep algorithm. Two-level generative model recovers the latent vertical/horizontal orientation and individual bar latents from pixels.

4x4 binary images. Top unit: vertical (p=2/3) vs. horizontal (p=1/3). Each of 4 candidate bars then on with p=0.2; pixels follow. 16 input, 8 first-hidden, 1 second-hidden net; 2x10^6 samples; final asymmetric KL = 0.10 bits.
