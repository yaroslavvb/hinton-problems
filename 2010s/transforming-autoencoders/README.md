# Transforming autoencoders (capsule precursor)

**Source:** Hinton, Krizhevsky & Wang (2011), "Transforming auto-encoders", ICANN.
**Demonstrates:** Disentangling "what" (entity present) from "where" (instantiation parameters). The seminal capsules paper.

MNIST with synthetic affine transformations. Each training pair is (image, transformed_image, dx/dy or affine matrix). 30 capsules each with a 3-unit recognition layer producing (presence-prob, x, y), 128 generative ReLU units, 22x22 reconstruction patch.
