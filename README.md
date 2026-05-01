# hinton-problems

Stubs for the synthetic learning problems and toy datasets that appear in Geoffrey Hinton's experimental papers from 1981 through 2022.

Each problem lives in its own folder containing:
- `README.md` — source paper, brief description, what it demonstrates
- `problem.py` — skeleton for dataset generation + model + training

Folders are grouped by decade. The catalog focuses on **synthetic toy problems** Hinton (or close collaborators) designed to isolate a representational property — the lineage from the 4-2-4 encoder (1985) through the shifter (1986), bars (1995), MultiMNIST (2017), Constellations (2019), Ellipse World (2022), and the Forward-Forward suite (2022).

## Catalog

### 1980s — Connectionist foundations
- [encoder-3-parity](1980s/encoder-3-parity/) — 3-bit even-parity, visible-only Boltzmann (Ackley, Hinton & Sejnowski 1985)
- [encoder-4-2-4](1980s/encoder-4-2-4/) — 2-bit Boltzmann bottleneck (1985)
- [encoder-4-3-4](1980s/encoder-4-3-4/) — over-complete error-correcting code (1985)
- [encoder-8-3-8](1980s/encoder-8-3-8/) — theoretical-minimum hidden capacity (1985)
- [encoder-40-10-40](1980s/encoder-40-10-40/) — large-scale Boltzmann encoder (1985)
- [shifter](1980s/shifter/) — shift-direction inference (Hinton & Sejnowski 1986)
- [grapheme-sememe](1980s/grapheme-sememe/) — synthetic word reading + lesion / relearning (1986)
- [xor](1980s/xor/) — canonical 2-bit XOR backprop (Rumelhart, Hinton & Williams 1986)
- [n-bit-parity](1980s/n-bit-parity/) — N-bit parity backprop (1986)
- [encoder-backprop-8-3-8](1980s/encoder-backprop-8-3-8/) — backprop encoder (1986)
- [distributed-to-local-bottleneck](1980s/distributed-to-local-bottleneck/) — graded single-unit bottleneck (1986)
- [symmetry](1980s/symmetry/) — 6-bit palindrome detection (1986)
- [binary-addition](1980s/binary-addition/) — two 2-bit numbers, local-minima study (1986)
- [negation](1980s/negation/) — flag-conditioned bit-flip (1986)
- [t-c-discrimination](1980s/t-c-discrimination/) — shared-weight retina (1986)
- [recurrent-shift-register](1980s/recurrent-shift-register/) — RNN learning a pure shift register (1986)
- [sequence-lookup-25](1980s/sequence-lookup-25/) — 25-sequence RNN look-up (1986)
- [family-trees](1980s/family-trees/) — kinship relations, two isomorphic trees (Hinton 1986)
- [riser-spectrogram](1980s/riser-spectrogram/) — synthetic riser/non-riser discrimination (Plaut & Hinton 1987)
- [fast-weights-rehearsal](1980s/fast-weights-rehearsal/) — two-time-scale weights with rehearsal (Hinton & Plaut 1987)

### 1990s — Unsupervised learning, mixtures, the Helmholtz machine
- [vowel-mixture-experts](1990s/vowel-mixture-experts/) — Peterson-Barney 4-class vowels (Jacobs, Jordan, Nowlan & Hinton 1991)
- [random-dot-stereograms](1990s/random-dot-stereograms/) — Imax / spatial coherence (Becker & Hinton 1992)
- [sunspots](1990s/sunspots/) — soft weight-sharing on Wolfer counts (Nowlan & Hinton 1992)
- [spline-images-factorial-vq](1990s/spline-images-factorial-vq/) — intrinsic-dim 5 curves (Hinton & Zemel 1994)
- [dipole-position](1990s/dipole-position/) — 8x8 dipole at random (x, y) (Zemel & Hinton 1995)
- [dipole-3d-constraint](1990s/dipole-3d-constraint/) — 3D constraint surface (1995)
- [dipole-what-where](1990s/dipole-what-where/) — discontinuous what/where bars (1995)
- [helmholtz-shifter](1990s/helmholtz-shifter/) — two-stage generative shifter (Dayan, Hinton, Neal & Zemel 1995)
- [bars](1990s/bars/) — 4x4 horizontal/vertical bars (Hinton, Dayan, Frey & Neal 1995)

### 2000s — Products of experts, contrastive divergence, deep belief nets
- [bars-rbm](2000s/bars-rbm/) — bars problem for RBM training (Hinton 2000)
- [transforming-pairs](2000s/transforming-pairs/) — gated conditional RBM (Memisevic & Hinton 2007)
- [bouncing-balls-2](2000s/bouncing-balls-2/) — TRBM video (Sutskever & Hinton 2007)
- [bouncing-balls-3](2000s/bouncing-balls-3/) — RTRBM 30x30 video (Sutskever, Hinton & Taylor 2008)

### 2010s — Capsules, distillation, attention
- [transforming-autoencoders](2010s/transforming-autoencoders/) — MNIST + affine, the seed of capsules (Hinton, Krizhevsky & Wang 2011)
- [deep-lambertian-spheres](2010s/deep-lambertian-spheres/) — synthetic spheres under multiple lighting (Tang, Salakhutdinov & Hinton 2012)
- [rnn-pathological](2010s/rnn-pathological/) — Hochreiter-Schmidhuber long-term-dep tasks (Sutskever, Martens, Dahl & Hinton 2013)
- [distillation-mnist-omitted-3](2010s/distillation-mnist-omitted-3/) — student never sees a "3" (Hinton, Vinyals & Dean 2015)
- [air-multimnist](2010s/air-multimnist/) — variable-count MNIST scenes (Eslami et al. 2016)
- [air-3d-primitives](2010s/air-3d-primitives/) — invert programmable renderer (2016)
- [fast-weights-associative-retrieval](2010s/fast-weights-associative-retrieval/) — c9k8j3f1??c -> 9 (Ba et al. 2016)
- [multi-level-glimpse-mnist](2010s/multi-level-glimpse-mnist/) — 24 hierarchical glimpses (2016)
- [catch-game](2010s/catch-game/) — partial-observability paddle/ball (2016)
- [multimnist-capsnet](2010s/multimnist-capsnet/) — overlapping digit pairs (Sabour, Frosst & Hinton 2017)
- [affnist](2010s/affnist/) — train on translated MNIST, test on affNIST (2017)
- [smallnorb-novel-viewpoint](2010s/smallnorb-novel-viewpoint/) — held-out azimuth / elevation (Hinton, Sabour & Frosst 2018)
- [constellations](2010s/constellations/) — 2D point-cloud part-whole grouping (Kosiorek, Sabour, Teh & Hinton 2019)

### 2020s — Subclass distillation, GLOM, Forward-Forward
- [mnist-2x5-subclass](2020s/mnist-2x5-subclass/) — super-class teacher with hidden subclass logits (Müller, Kornblith & Hinton 2020)
- [geo-flow-capsules](2020s/geo-flow-capsules/) — Geo / Geo+ moving 2D shapes (Sabour, Tagliasacchi, Yazdani, Hinton & Fleet 2021)
- [ellipse-world](2020s/ellipse-world/) — eGLOM ambiguous-part-to-whole (Culp, Sabour & Hinton 2022)
- [ff-hybrid-mnist](2020s/ff-hybrid-mnist/) — hand-crafted hybrid-image MNIST negatives (Hinton 2022)
- [ff-label-in-input](2020s/ff-label-in-input/) — first 10 pixels carry one-hot label (2022)
- [ff-recurrent-mnist](2020s/ff-recurrent-mnist/) — repeated-frame "video" with top-down recurrence (2022)
- [ff-cifar-locally-connected](2020s/ff-cifar-locally-connected/) — CIFAR-10 with locally-connected FF (2022)
- [ff-aesop-sequences](2020s/ff-aesop-sequences/) — Aesop's Fables next-character with self-generated negatives (2022)

## Structure

```
problem-folder/
├── README.md      one paragraph: source + property
└── problem.py     stubs: generate_dataset, build_model, train
```

The stubs raise `NotImplementedError`. Fill in the parts you need.
