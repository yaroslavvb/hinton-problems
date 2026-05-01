# Ellipse World

**Source:** Culp, Sabour & Hinton (2022), "Testing GLOM's ability to infer wholes from ambiguous parts".
**Demonstrates:** eGLOM — an MLP-replicated-per-location architecture with transformer-style attention within object level — can infer wholes (face, sheep, ...) from highly ambiguous individual ellipses by exploiting spatial relationships.

Each "image" is a 2D grid; some cells contain a single 6-DoF ellipse symbol; objects are composed of exactly 5 ellipses arranged into class-defining configurations via a global affine.
