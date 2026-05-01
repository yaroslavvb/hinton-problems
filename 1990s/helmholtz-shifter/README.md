# Helmholtz shifter

**Source:** Dayan, Hinton, Neal & Zemel (1995), "The Helmholtz machine", Neural Computation.
**Demonstrates:** Two-stage generative process (direction -> bottom-row pixels). After training, layer-3 units become shift-direction units; layer-2 units detect specific shifted bit-pairs. Analogous to extracting depth from stereo.

4x8 binary patterns; bottom row random binary, top row left/right-shifted (with wraparound) copy; middle two rows duplicate outer rows.
