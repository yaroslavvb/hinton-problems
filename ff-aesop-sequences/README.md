# Forward-Forward: Aesop's Fables next-character

**Source:** Hinton (2022).
**Demonstrates:** Two FF variants — teacher-forcing negatives (positive = real 10-char string; negative = same first 9 chars + a model-predicted incorrect next char) and self-generated offline negatives (entirely model-generated autoregressively from initial 10 chars). Both work nearly identically — a key result because it means negative passes can be performed offline during a "sleep phase", decoupling positive (wake) and negative (sleep) learning.

248 strings x 100 chars each, 30-symbol alphabet (lowercase + space + comma + semicolon + full stop); first 10 chars given as context, predict the next 90. Three-hidden-layer net (each layer 2000 ReLUs).
