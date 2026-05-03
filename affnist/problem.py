"""
Train on translated MNIST, test on affNIST (Sabour, Frosst & Hinton 2017,
"Dynamic routing between capsules"). The full implementation lives in
`affnist.py`; this file forwards the spec-required entry points so
`from problem import *` is enough to import the working pieces.
"""
from affnist import (
    make_translated_mnist,
    load_affnist_test,
    evaluate_robustness,
    train,
    evaluate,
    TinyCapsNet,
    TinyCNN,
    load_mnist,
    synthesize_affnist,
)


if __name__ == "__main__":
    from affnist import main
    main()
