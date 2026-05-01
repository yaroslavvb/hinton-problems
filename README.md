# hinton-problems

Stubs for the synthetic learning problems and toy datasets that appear in Geoffrey Hinton's experimental papers from 1981 through 2022.

Each problem lives in its own folder containing:
- `README.md` — source paper, brief description, what it demonstrates
- `problem.py` — skeleton for dataset generation + model + training

The catalog focuses on **synthetic toy problems** Hinton (or close collaborators) designed to isolate a representational property — the lineage from the 4-2-4 encoder (1985) through the shifter (1986), bars (1995), MultiMNIST (2017), Constellations (2019), Ellipse World (2022), and the Forward-Forward suite (2022). Folders are flat; the catalog below is grouped by year for readability.

## Catalog

### 1980s — Connectionist foundations

**Ackley, Hinton & Sejnowski (1985)** — A learning algorithm for Boltzmann machines
- [encoder-3-parity](encoder-3-parity/) — 3-bit even-parity, visible-only Boltzmann
- [encoder-4-2-4](encoder-4-2-4/) — 2-bit Boltzmann bottleneck
- [encoder-4-3-4](encoder-4-3-4/) — over-complete error-correcting code
- [encoder-8-3-8](encoder-8-3-8/) — theoretical-minimum hidden capacity
- [encoder-40-10-40](encoder-40-10-40/) — large-scale Boltzmann encoder

**Hinton & Sejnowski (1986)** — Learning and relearning in Boltzmann machines
- [shifter](shifter/) — shift-direction inference (the canonical higher-order-feature toy)
- [grapheme-sememe](grapheme-sememe/) — synthetic word reading + lesion / relearning

**Rumelhart, Hinton & Williams (1986)** — Learning internal representations by error propagation
- [xor](xor/) — canonical 2-bit XOR backprop
- [n-bit-parity](n-bit-parity/) — N-bit parity backprop
- [encoder-backprop-8-3-8](encoder-backprop-8-3-8/) — backprop encoder
- [distributed-to-local-bottleneck](distributed-to-local-bottleneck/) — graded single-unit bottleneck
- [symmetry](symmetry/) — 6-bit palindrome detection
- [binary-addition](binary-addition/) — two 2-bit numbers, local-minima study
- [negation](negation/) — flag-conditioned bit-flip
- [t-c-discrimination](t-c-discrimination/) — shared-weight retina
- [recurrent-shift-register](recurrent-shift-register/) — RNN learning a pure shift register
- [sequence-lookup-25](sequence-lookup-25/) — 25-sequence RNN look-up

**Hinton (1986)** — Learning distributed representations of concepts
- [family-trees](family-trees/) — kinship relations, two isomorphic trees

**Plaut & Hinton (1987)** — Learning sets of filters using back-propagation
- [riser-spectrogram](riser-spectrogram/) — synthetic riser/non-riser discrimination

**Hinton & Plaut (1987)** — Using fast weights to deblur old memories
- [fast-weights-rehearsal](fast-weights-rehearsal/) — two-time-scale weights with rehearsal

### 1990s — Unsupervised learning, mixtures, the Helmholtz machine

**Jacobs, Jordan, Nowlan & Hinton (1991)** — Adaptive mixtures of local experts
- [vowel-mixture-experts](vowel-mixture-experts/) — Peterson-Barney 4-class vowels

**Becker & Hinton (1992)** — A self-organizing neural network that discovers surfaces in random-dot stereograms
- [random-dot-stereograms](random-dot-stereograms/) — Imax / spatial coherence

**Nowlan & Hinton (1992)** — Simplifying neural networks by soft weight-sharing
- [sunspots](sunspots/) — soft weight-sharing on Wolfer counts

**Hinton & Zemel (1994)** — Autoencoders, MDL and Helmholtz free energy
- [spline-images-factorial-vq](spline-images-factorial-vq/) — intrinsic-dim 5 curves

**Zemel & Hinton (1995)** — Learning population codes by minimizing description length
- [dipole-position](dipole-position/) — 8x8 dipole at random (x, y)
- [dipole-3d-constraint](dipole-3d-constraint/) — 3D constraint surface
- [dipole-what-where](dipole-what-where/) — discontinuous what/where bars

**Dayan, Hinton, Neal & Zemel (1995)** — The Helmholtz machine
- [helmholtz-shifter](helmholtz-shifter/) — two-stage generative shifter

**Hinton, Dayan, Frey & Neal (1995)** — The wake-sleep algorithm
- [bars](bars/) — 4x4 horizontal/vertical bars

### 2000s — Products of experts, contrastive divergence, deep belief nets

**Hinton (2000)** — Training products of experts by minimizing contrastive divergence
- [bars-rbm](bars-rbm/) — bars problem for RBM training

**Memisevic & Hinton (2007)** — Unsupervised learning of image transformations
- [transforming-pairs](transforming-pairs/) — gated conditional RBM

**Sutskever & Hinton (2007)** — Learning multilevel distributed representations for high-dimensional sequences
- [bouncing-balls-2](bouncing-balls-2/) — TRBM video

**Sutskever, Hinton & Taylor (2008)** — The recurrent temporal RBM
- [bouncing-balls-3](bouncing-balls-3/) — RTRBM 30x30 video

### 2010s — Capsules, distillation, attention

**Hinton, Krizhevsky & Wang (2011)** — Transforming auto-encoders
- [transforming-autoencoders](transforming-autoencoders/) — MNIST + affine, the seed of capsules

**Tang, Salakhutdinov & Hinton (2012)** — Deep Lambertian Networks
- [deep-lambertian-spheres](deep-lambertian-spheres/) — synthetic spheres under multiple lighting

**Sutskever, Martens, Dahl & Hinton (2013)** — On the importance of initialization and momentum
- [rnn-pathological](rnn-pathological/) — Hochreiter-Schmidhuber long-term-dep tasks

**Hinton, Vinyals & Dean (2015)** — Distilling the knowledge in a neural network
- [distillation-mnist-omitted-3](distillation-mnist-omitted-3/) — student never sees a "3"

**Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu & Hinton (2016)** — Attend, Infer, Repeat
- [air-multimnist](air-multimnist/) — variable-count MNIST scenes
- [air-3d-primitives](air-3d-primitives/) — invert programmable renderer

**Ba, Hinton, Mnih, Leibo & Ionescu (2016)** — Using fast weights to attend to the recent past
- [fast-weights-associative-retrieval](fast-weights-associative-retrieval/) — `c9k8j3f1??c -> 9`
- [multi-level-glimpse-mnist](multi-level-glimpse-mnist/) — 24 hierarchical glimpses
- [catch-game](catch-game/) — partial-observability paddle/ball

**Sabour, Frosst & Hinton (2017)** — Dynamic routing between capsules
- [multimnist-capsnet](multimnist-capsnet/) — overlapping digit pairs
- [affnist](affnist/) — train on translated MNIST, test on affNIST

**Hinton, Sabour & Frosst (2018)** — Matrix capsules with EM routing
- [smallnorb-novel-viewpoint](smallnorb-novel-viewpoint/) — held-out azimuth / elevation

**Kosiorek, Sabour, Teh & Hinton (2019)** — Stacked capsule autoencoders
- [constellations](constellations/) — 2D point-cloud part-whole grouping

### 2020s — Subclass distillation, GLOM, Forward-Forward

**Müller, Kornblith & Hinton (2020)** — Subclass distillation
- [mnist-2x5-subclass](mnist-2x5-subclass/) — super-class teacher with hidden subclass logits

**Sabour, Tagliasacchi, Yazdani, Hinton & Fleet (2021)** — Unsupervised part representation by flow capsules
- [geo-flow-capsules](geo-flow-capsules/) — Geo / Geo+ moving 2D shapes

**Culp, Sabour & Hinton (2022)** — Testing GLOM's ability to infer wholes from ambiguous parts
- [ellipse-world](ellipse-world/) — eGLOM ambiguous-part-to-whole

**Hinton (2022)** — The forward-forward algorithm: some preliminary investigations
- [ff-hybrid-mnist](ff-hybrid-mnist/) — hand-crafted hybrid-image MNIST negatives
- [ff-label-in-input](ff-label-in-input/) — first 10 pixels carry one-hot label
- [ff-recurrent-mnist](ff-recurrent-mnist/) — repeated-frame "video" with top-down recurrence
- [ff-cifar-locally-connected](ff-cifar-locally-connected/) — CIFAR-10 with locally-connected FF
- [ff-aesop-sequences](ff-aesop-sequences/) — Aesop's Fables next-character with self-generated negatives

## Structure

```
problem-folder/
├── README.md      one paragraph: source + property
└── problem.py     stubs: generate_dataset, build_model, train
```

The stubs raise `NotImplementedError`. Fill in the parts you need.
