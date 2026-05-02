"""
Family-trees / kinship task — Hinton (1986), "Learning distributed
representations of concepts", Cognitive Science 8.

Two isomorphic 12-person family trees (English + Italian) and 12 kinship
relations. The network learns to predict P2 from (P1, R). The 6-unit
"person encoding" layer self-organizes into interpretable axes for
nationality, generation, and family branch.

Architecture (Hinton 1986, Fig 2):

    24-unit one-hot person  ->  6-unit person encoding  -+
                                                          \
                                                           +-> 12-unit
    12-unit one-hot relation -> 6-unit relation encoding -+    central
                                                               |
                                                               v
                                       24-unit output <- 6-unit decoder

Training: full-batch backprop with logistic (sigmoid) hidden units,
softmax + cross-entropy at the 24-way output, plus momentum and a small
weight decay to keep activations on-manifold.

Hinton's 1986 paper used a "quadratic error measure" on sigmoid outputs;
modern reproductions almost universally swap that for softmax + CE because
the one-hot 24-class target gives the squared-error loss a vanishingly
small positive-class signal vs. 23 zero-target push-down terms. The
hidden units are still logistic. See "Deviations from the original" in
the README.
"""

from __future__ import annotations
import argparse
import time
from typing import Any
import numpy as np


# ----------------------------------------------------------------------
# People, relations, and per-person attributes
# ----------------------------------------------------------------------

ENGLISH_PEOPLE = [
    "Christopher", "Penelope", "Andrew", "Christine",
    "Margaret", "Arthur", "Victoria", "James",
    "Jennifer", "Charles", "Colin", "Charlotte",
]
ITALIAN_PEOPLE = [
    "Roberto", "Maria", "Pierro", "Francesca",
    "Gina", "Emilio", "Lucia", "Marco",
    "Angela", "Tomaso", "Alfonso", "Sophia",
]
RELATIONS = [
    "father", "mother", "husband", "wife",
    "son", "daughter", "uncle", "aunt",
    "brother", "sister", "nephew", "niece",
]

ALL_PEOPLE = ENGLISH_PEOPLE + ITALIAN_PEOPLE  # 24
PERSON_INDEX = {p: i for i, p in enumerate(ALL_PEOPLE)}
RELATION_INDEX = {r: i for i, r in enumerate(RELATIONS)}

# Sex of each person (M / F). Italians mirror English position-by-position.
SEX_OF: dict[str, str] = {
    "Christopher": "M", "Penelope": "F", "Andrew": "M", "Christine": "F",
    "Margaret": "F", "Arthur": "M", "Victoria": "F", "James": "M",
    "Jennifer": "F", "Charles": "M", "Colin": "M", "Charlotte": "F",
    "Roberto": "M", "Maria": "F", "Pierro": "M", "Francesca": "F",
    "Gina": "F", "Emilio": "M", "Lucia": "F", "Marco": "M",
    "Angela": "F", "Tomaso": "M", "Alfonso": "M", "Sophia": "F",
}

# Per-tree position (0..11) attributes, identical between the two trees:
#   pos 0..3   = generation 1 (4 grandparents)
#   pos 4..9   = generation 2 (3 couples = 6 people)
#   pos 10..11 = generation 3 (2 grandchildren)
# In the tree: positions 0/1 are the "Christopher / Roberto" branch couple,
# 2/3 are the "Andrew / Pierro" branch couple, 4/5 (Margaret/Arthur) are the
# cross-tree gen-2 couple whose children are 10/11. 6/7 (Victoria/James) and
# 8/9 (Jennifer/Charles) are the other two gen-2 couples.
_GEN_BY_POS = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
# Branch within a tree:
#   0 = "left-grandparent" branch (Christopher / Roberto blood line)
#   1 = "right-grandparent" branch (Andrew / Pierro blood line)
#   2 = no parents in tree (the two outsiders married into the tree:
#       James / Marco at pos 7, Charles / Tomaso at pos 9).
#   Mixed-blood gen-3 children (Colin/Alfonso, Charlotte/Sophia) are coded
#   by their FATHER's branch (Arthur is left-branch).
_BRANCH_BY_POS = [0, 0, 1, 1, 1, 0, 0, 2, 1, 2, 0, 0]


def attribute_table() -> dict[str, np.ndarray]:
    """Return per-person {nationality, generation, branch} arrays of shape (24,).

    nationality: 0 = English, 1 = Italian
    generation : 1 / 2 / 3
    branch     : 0 / 1 / 2 (see `_BRANCH_BY_POS` above)
    """
    nationality = np.array([0] * 12 + [1] * 12, dtype=np.int32)
    generation = np.array(_GEN_BY_POS * 2, dtype=np.int32)
    branch = np.array(_BRANCH_BY_POS * 2, dtype=np.int32)
    return {"nationality": nationality, "generation": generation, "branch": branch}


# ----------------------------------------------------------------------
# Build the family-tree relation triples
# ----------------------------------------------------------------------

def _build_one_tree(people: list[str]) -> list[tuple[str, str, str]]:
    """Generate every valid (P, R, target) triple for one 12-person tree.

    Tree structure (Hinton 1986):

        Christopher = Penelope         Andrew = Christine
                |                              |
        +-------+-------+              +-------+-------+
        |               |              |               |
    Arthur = Margaret(*) Victoria = James     Jennifer = Charles
            |
        +---+---+
        |       |
      Colin  Charlotte

    (*) Cross-tree marriage: Arthur is Christopher/Penelope's son and Margaret
    is Andrew/Christine's daughter. James and Charles are outsiders married in.
    Italian tree mirrors English position-by-position.
    """
    P = people
    assert len(P) == 12

    spouse_pairs = [(0, 1), (2, 3), (5, 4), (6, 7), (8, 9)]
    children_of_couple = {
        (0, 1): [5, 6],     # Christopher, Penelope -> Arthur, Victoria
        (2, 3): [4, 8],     # Andrew, Christine     -> Margaret, Jennifer
        (5, 4): [10, 11],   # Arthur, Margaret      -> Colin, Charlotte
    }

    spouse: dict[str, str] = {}
    for a, b in spouse_pairs:
        spouse[P[a]] = P[b]
        spouse[P[b]] = P[a]

    children: dict[str, list[str]] = {p: [] for p in P}
    parents: dict[str, list[str]] = {p: [] for p in P}
    for (a, b), kids in children_of_couple.items():
        for k in kids:
            children[P[a]].append(P[k])
            children[P[b]].append(P[k])
            parents[P[k]].append(P[a])
            parents[P[k]].append(P[b])

    def siblings_of(person: str) -> list[str]:
        ps = set(parents[person])
        if not ps:
            return []
        return [q for q in P if q != person and set(parents[q]) == ps]

    triples: list[tuple[str, str, str]] = []

    for person in P:
        # spouse -> husband / wife (target's sex determines the relation label)
        if person in spouse:
            sp = spouse[person]
            rel = "husband" if SEX_OF[sp] == "M" else "wife"
            triples.append((person, rel, sp))
        # parents -> father / mother
        for par in parents[person]:
            rel = "father" if SEX_OF[par] == "M" else "mother"
            triples.append((person, rel, par))
        # children -> son / daughter
        for ch in children[person]:
            rel = "son" if SEX_OF[ch] == "M" else "daughter"
            triples.append((person, rel, ch))
        # siblings -> brother / sister
        for sib in siblings_of(person):
            rel = "brother" if SEX_OF[sib] == "M" else "sister"
            triples.append((person, rel, sib))
        # uncle / aunt: sibling of a parent (blood) plus that sibling's spouse
        for par in parents[person]:
            for par_sib in siblings_of(par):
                rel = "uncle" if SEX_OF[par_sib] == "M" else "aunt"
                triples.append((person, rel, par_sib))
                if par_sib in spouse:
                    sp_of = spouse[par_sib]
                    rel2 = "uncle" if SEX_OF[sp_of] == "M" else "aunt"
                    triples.append((person, rel2, sp_of))
        # nephew / niece: child of a sibling, plus child of spouse's sibling
        nephew_pool: list[str] = []
        for sib in siblings_of(person):
            nephew_pool.extend(children[sib])
        if person in spouse:
            for sib in siblings_of(spouse[person]):
                nephew_pool.extend(children[sib])
        for n in nephew_pool:
            rel = "nephew" if SEX_OF[n] == "M" else "niece"
            triples.append((person, rel, n))

    return sorted(set(triples))


def build_triples() -> list[tuple[str, str, str]]:
    """Return every valid (person1, relation, person2) triple across both trees."""
    return _build_one_tree(ENGLISH_PEOPLE) + _build_one_tree(ITALIAN_PEOPLE)


def aggregate_facts(triples) -> list[tuple[str, str, list[str]]]:
    """Group (P1, R, P2) triples by (P1, R) -> sorted list of P2.

    Many relations are many-to-one (e.g., Christopher has two children, so
    `(Christopher, daughter, Victoria)` is one triple but `(Andrew,
    daughter, ?)` answers `{Margaret, Jennifer}`). Hinton's 1986 setup
    treats one (P1, R) as one training example with a multi-hot target;
    this aggregation matches that.
    """
    from collections import defaultdict
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p1, r, p2 in triples:
        groups[(p1, r)].append(p2)
    return sorted([(p1, r, sorted(t)) for (p1, r), t in groups.items()])


def split_train_test(facts, n_test: int = 4, seed: int = 0):
    """Random hold-out split over facts. Returns (train_list, test_list).

    `facts` is the output of `aggregate_facts` -- a list of
    (P1, R, [P2,...]) tuples. Holding out at the fact level means the
    network never sees the (P1, R) pair during training; it must
    generalize from related facts (the headline finding).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(facts))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return [facts[i] for i in train_idx], [facts[i] for i in test_idx]


def facts_to_arrays(facts) -> tuple[np.ndarray, np.ndarray]:
    """Encode facts as ((n,36) one-hot input, (n,24) soft-distribution target).

    Multi-valued facts (e.g., Andrew has two daughters) become uniform
    distributions over the valid answers. Single-valued facts collapse to
    one-hot. This is what softmax + cross-entropy expects -- the gradient
    splits evenly across the valid targets.
    """
    n = len(facts)
    X = np.zeros((n, 24 + 12), dtype=np.float32)
    Y = np.zeros((n, 24), dtype=np.float32)
    for i, (p1, r, targets) in enumerate(facts):
        X[i, PERSON_INDEX[p1]] = 1.0
        X[i, 24 + RELATION_INDEX[r]] = 1.0
        w = 1.0 / len(targets)
        for t in targets:
            Y[i, PERSON_INDEX[t]] = w
    return X, Y


# Backwards-compatible alias used by the stub-style API.
def triples_to_arrays(triples_or_facts) -> tuple[np.ndarray, np.ndarray]:
    """If passed raw triples, aggregate first; if facts, use directly."""
    if not triples_or_facts:
        return np.zeros((0, 36), np.float32), np.zeros((0, 24), np.float32)
    first = triples_or_facts[0]
    if isinstance(first[2], list):
        return facts_to_arrays(triples_or_facts)
    return facts_to_arrays(aggregate_facts(triples_or_facts))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def build_model(init_scale: float = 1.0, seed: int = 0) -> "FamilyTreeMLP":
    """Public constructor matching the stub signature."""
    return FamilyTreeMLP(init_scale=init_scale, seed=seed)


class FamilyTreeMLP:
    """24-input + 12-relation -> two 6-unit encoders -> 12 -> 6 -> 24 output.

    `tanh` hidden units, softmax output. Hinton (1986) used logistic units
    throughout; we use tanh so that a four-layer chain of squashing
    nonlinearities still passes a usable gradient back to the person
    encoder. (sigmoid'(0) = 0.25 vs tanh'(0) = 1; the four-layer product
    is 0.0039 vs 1.0.) The 6-unit person-encoding layer is what we
    interrogate for interpretable axes -- still single-sided like a
    logistic encoding once tanh is squashed and Xavier-scaled, see the
    visualization scripts.
    """

    LAYER_PARAMS = ("W_p", "b_p", "W_r", "b_r",
                    "W_c", "b_c", "W_d", "b_d", "W_o", "b_o")

    def __init__(self, init_scale: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)

        def xav(n_in: int, n_out: int) -> np.ndarray:
            sigma = init_scale * np.sqrt(2.0 / (n_in + n_out))
            return (sigma * rng.standard_normal((n_in, n_out))).astype(np.float32)

        # Person encoder: 24 -> 6
        self.W_p = xav(24, 6)
        self.b_p = np.zeros(6, dtype=np.float32)
        # Relation encoder: 12 -> 6
        self.W_r = xav(12, 6)
        self.b_r = np.zeros(6, dtype=np.float32)
        # Central: (6+6)=12 -> 12
        self.W_c = xav(12, 12)
        self.b_c = np.zeros(12, dtype=np.float32)
        # Decoder: 12 -> 6
        self.W_d = xav(12, 6)
        self.b_d = np.zeros(6, dtype=np.float32)
        # Output projection: 6 -> 24 (softmax over the 24 people)
        self.W_o = xav(6, 24)
        self.b_o = np.zeros(24, dtype=np.float32)

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        x_p = X[:, :24]
        x_r = X[:, 24:]
        h_p = np.tanh(x_p @ self.W_p + self.b_p)
        h_r = np.tanh(x_r @ self.W_r + self.b_r)
        h_concat = np.concatenate([h_p, h_r], axis=1)
        h_c = np.tanh(h_concat @ self.W_c + self.b_c)
        h_d = np.tanh(h_c @ self.W_d + self.b_d)
        z_o = h_d @ self.W_o + self.b_o
        y = softmax(z_o)
        return {"x_p": x_p, "x_r": x_r,
                "h_p": h_p, "h_r": h_r,
                "h_concat": h_concat, "h_c": h_c, "h_d": h_d, "y": y}

    def predict_argmax(self, X: np.ndarray) -> np.ndarray:
        """Single-best prediction via argmax over the 24 output units."""
        return np.argmax(self.forward(X)["y"], axis=1)

    def predict_topk(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        """Indices of the top-k output units, sorted by descending probability."""
        y = self.forward(X)["y"]
        return np.argsort(-y, axis=1)[:, :k]

    # Backwards-compatible alias used by some helpers / external callers.
    predict = predict_argmax


def backward(model: FamilyTreeMLP, cache: dict, Y: np.ndarray) -> dict:
    """Backprop through tanh hidden units with softmax + cross-entropy output.

    `Y` is the soft target distribution: a row sums to 1 and has uniform mass
    over each fact's set of valid answers. The gradient at the output is the
    standard `(softmax - Y)` form.
    """
    n = Y.shape[0]
    y = cache["y"]
    dz_o = (y - Y) / n
    dW_o = cache["h_d"].T @ dz_o
    db_o = dz_o.sum(axis=0)
    dh_d = dz_o @ model.W_o.T
    dz_d = dh_d * (1.0 - cache["h_d"] ** 2)
    dW_d = cache["h_c"].T @ dz_d
    db_d = dz_d.sum(axis=0)
    dh_c = dz_d @ model.W_d.T
    dz_c = dh_c * (1.0 - cache["h_c"] ** 2)
    dW_c = cache["h_concat"].T @ dz_c
    db_c = dz_c.sum(axis=0)
    dh_concat = dz_c @ model.W_c.T
    dh_p = dh_concat[:, :6]
    dh_r = dh_concat[:, 6:]
    dz_p = dh_p * (1.0 - cache["h_p"] ** 2)
    dz_r = dh_r * (1.0 - cache["h_r"] ** 2)
    dW_p = cache["x_p"].T @ dz_p
    db_p = dz_p.sum(axis=0)
    dW_r = cache["x_r"].T @ dz_r
    db_r = dz_r.sum(axis=0)
    return {"W_p": dW_p, "b_p": db_p,
            "W_r": dW_r, "b_r": db_r,
            "W_c": dW_c, "b_c": db_c,
            "W_d": dW_d, "b_d": db_d,
            "W_o": dW_o, "b_o": db_o}


def loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Cross-entropy with soft targets, mean over the batch."""
    eps = 1e-12
    return float(-np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)))


def accuracy(model: FamilyTreeMLP, X: np.ndarray, Y: np.ndarray) -> float:
    """Fraction of facts whose argmax prediction lies in the valid target set.

    Multi-valued facts (Andrew has two daughters) are correct as long as the
    network's argmax is one of the valid answers. This matches Hinton's
    evaluation in the 1986 paper -- credit for picking *any* correct P2.
    """
    if len(X) == 0:
        return float("nan")
    pred = model.predict_argmax(X)
    target = (Y > 0).astype(np.int32)
    correct = target[np.arange(len(pred)), pred] > 0
    return float(np.mean(correct))


def per_unit_accuracy(model: FamilyTreeMLP, X: np.ndarray, Y: np.ndarray,
                      threshold: float = 0.05) -> float:
    """Fraction of facts whose top-k argmax exactly recovers every valid target.

    With softmax outputs each fact's targets share a probability budget of 1,
    so the cleanest "all answers" check is to take the top |targets| outputs
    and require set equality. This is the strict version of `accuracy`.
    """
    if len(X) == 0:
        return float("nan")
    y = model.forward(X)["y"]
    target = (Y > 0).astype(np.int32)
    n_targets = target.sum(axis=1)
    matches = np.zeros(len(X), dtype=bool)
    for i, k in enumerate(n_targets):
        topk = np.argsort(-y[i])[:int(k)]
        pred_set = np.zeros(24, dtype=np.int32)
        pred_set[topk] = 1
        matches[i] = np.array_equal(pred_set, target[i])
    return float(np.mean(matches))


def train(model: FamilyTreeMLP,
          X_train: np.ndarray, Y_train: np.ndarray,
          X_test: np.ndarray | None = None,
          Y_test: np.ndarray | None = None,
          n_sweeps: int = 10000,
          lr: float = 0.5,
          momentum: float = 0.9,
          weight_decay: float = 0.0,
          snapshot_callback=None,
          snapshot_every: int = 25,
          verbose: bool = True) -> dict:
    """Full-batch backprop with momentum.

    The relatively large `lr` partly compensates for the (1/n) factor in the
    backward pass; momentum + a small weight decay keep the trajectory stable.
    """
    history: dict[str, list[Any]] = {
        "epoch": [], "train_loss": [], "train_acc": [], "test_acc": []
    }

    velocities = {k: np.zeros_like(getattr(model, k))
                  for k in FamilyTreeMLP.LAYER_PARAMS}

    for epoch in range(n_sweeps):
        cache = model.forward(X_train)
        l = loss(cache["y"], Y_train)
        grads = backward(model, cache, Y_train)
        for k, g in grads.items():
            param = getattr(model, k)
            wd = weight_decay if k.startswith("W") else 0.0
            velocities[k] = momentum * velocities[k] + lr * (g + wd * param)
            setattr(model, k, param - velocities[k])

        train_acc = accuracy(model, X_train, Y_train)
        test_acc = (accuracy(model, X_test, Y_test)
                    if X_test is not None else float("nan"))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(l)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if verbose and (epoch % 100 == 0 or epoch == n_sweeps - 1):
            print(f"epoch {epoch+1:5d}  loss={l:.4f}  "
                  f"train_acc={train_acc*100:5.1f}%  "
                  f"test_acc={test_acc*100:5.1f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_sweeps - 1):
            snapshot_callback(epoch, model, history)

    return history


def inspect_person_encoding(model: FamilyTreeMLP) -> np.ndarray:
    """Return the 6-D person-encoding code for each of the 24 people.

    Output shape (24, 6); row i corresponds to ALL_PEOPLE[i].
    """
    X_p = np.eye(24, dtype=np.float32)
    return np.tanh(X_p @ model.W_p + model.b_p)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-test", type=int, default=4)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    triples = build_triples()
    facts = aggregate_facts(triples)
    print(f"# Total triples: {len(triples)} "
          f"(English: {len(_build_one_tree(ENGLISH_PEOPLE))}, "
          f"Italian: {len(_build_one_tree(ITALIAN_PEOPLE))})")
    print(f"# Total (P, R) facts: {len(facts)}")

    train_facts, test_facts = split_train_test(facts,
                                               n_test=args.n_test,
                                               seed=args.seed)
    print(f"# Train facts: {len(train_facts)}, Test facts: {len(test_facts)}")

    X_train, Y_train = facts_to_arrays(train_facts)
    X_test, Y_test = facts_to_arrays(test_facts)

    model = build_model(init_scale=args.init_scale, seed=args.seed)
    t0 = time.time()
    history = train(model, X_train, Y_train, X_test, Y_test,
                    n_sweeps=args.epochs, lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"\n# Trained in {elapsed:.1f}s")
    print(f"# Final train accuracy: {history['train_acc'][-1]*100:.1f}%")
    print(f"# Final test accuracy: {history['test_acc'][-1]*100:.1f}%")

    print("\n# Held-out test facts:")
    n_test_correct = 0
    y_test = model.forward(X_test)["y"]
    test_argmax = np.argmax(y_test, axis=1)
    for (p1, r, targets), idx, probs in zip(test_facts, test_argmax, y_test):
        guess = ALL_PEOPLE[idx]
        ok = guess in targets
        n_test_correct += int(ok)
        mark = "OK " if ok else "X  "
        # Top-2 to show what the network was considering
        top2 = np.argsort(-probs)[:2]
        runner = ALL_PEOPLE[top2[1]] if top2[1] != idx else ALL_PEOPLE[top2[0]]
        print(f"  {mark} ({p1:11s}, {r:8s}, ?) -> {guess:11s}  "
              f"[true: {sorted(targets)}]  "
              f"runner-up: {runner} ({probs[top2[1]]:.2f})")
    print(f"\n# Test correct (argmax in valid set): "
          f"{n_test_correct}/{len(test_facts)}")

    # Brief look at the 6-unit person encoding
    codes = inspect_person_encoding(model)
    attrs = attribute_table()
    print(f"\n# 6-D person encoding shape: {codes.shape}")
    nat_means = [codes[attrs['nationality'] == i].mean(axis=0)
                 for i in range(2)]
    diff = nat_means[0] - nat_means[1]
    print("# Per-unit mean difference English - Italian "
          "(large |diff| = unit codes nationality):")
    for j, d in enumerate(diff):
        print(f"  unit {j}: {d:+.3f}")


if __name__ == "__main__":
    main()
