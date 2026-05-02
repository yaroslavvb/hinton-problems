"""
25-sequence look-up RNN task (Rumelhart, Hinton & Williams 1986).

Thin compatibility shim: the actual implementation lives in
`sequence_lookup_25.py`. This module re-exports the public API so the
stub's original function names continue to work.
"""

from sequence_lookup_25 import (  # noqa: F401
    SequenceLookupRNN,
    build_rnn,
    generate_dataset,
    test_generalization,
    train,
    train_bptt,
)


if __name__ == "__main__":
    from sequence_lookup_25 import main
    main()
