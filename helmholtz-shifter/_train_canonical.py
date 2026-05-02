"""Run the canonical training and save weights + history + snapshots.

Used by both visualize_helmholtz_shifter.py and make_helmholtz_shifter_gif.py
so they share one trained network rather than re-training twice.

Saved artifacts (in `viz/`):
    canonical_model.npz      -- weights + biases of the trained machine
    canonical_history.npz    -- IS-NLL + dir-acc trajectory
    canonical_snapshots.npz  -- per-snapshot weights + fantasies for the GIF
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

import helmholtz_shifter as hs


def train_and_save(seed: int, n_passes: int, lr: float, batch_size: int,
                   eval_every: int, snapshot_every: int, outdir: str,
                   n_hidden: int = hs.N_HID_DEFAULT,
                   n_top: int = hs.N_TOP_DEFAULT,
                   p_on: float = hs.P_ON,
                   eval_n_samples: int = 50,
                   ) -> tuple[hs.HelmholtzMachine, dict, list]:
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(123)
    eval_set = hs.generate_shifter(256, p_on=p_on, rng=eval_rng)

    model = hs.HelmholtzMachine(seed=seed, n_hidden=n_hidden, n_top=n_top,
                                p_on=p_on)
    snapshots = []  # (step, W_hv, W_th, b_v, b_h, b_top, fantasy_v, nll, dir_acc)

    def cb(step, m, history):
        if step % snapshot_every == 0 or step == n_passes:
            v_fant, _, _ = m.generate(64)
            snapshots.append((
                step,
                m.W_hv.copy(), m.W_th.copy(),
                m.b_v.copy(), m.b_h.copy(), m.b_top.copy(),
                v_fant.copy(),
                float(history["is_nll_bits"][-1]),
                float(history["dir_acc"][-1]),
            ))
            print(f"  snapshot at step {step:>10d}  "
                  f"NLL={history['is_nll_bits'][-1]:.3f}  "
                  f"dir-acc={history['dir_acc'][-1]:.3f}")

    t0 = time.time()
    history = hs.wake_sleep(model, rng, n_passes=n_passes, lr=lr,
                            batch_size=batch_size,
                            eval_every=eval_every,
                            eval_callback=cb,
                            eval_set=eval_set,
                            eval_n_samples=eval_n_samples)
    elapsed = time.time() - t0

    # Final evaluation: more importance samples, larger fantasy set
    log_p = model.importance_log_prob(eval_set[0], n_samples=200,
                                      rng=eval_rng)
    final_nll = float(-log_p.mean() / np.log(2.0))
    inspect = hs.inspect_layer3_units(model, n_fantasy=2048, rng=eval_rng)
    final_dir_acc = hs.direction_recovery(model, eval_set[0], eval_set[1],
                                          n_draws=11, rng=eval_rng)
    print(f"Done. Final IS-NLL={final_nll:.4f}, dir-acc={final_dir_acc:.3f}, "
          f"best-unit-selectivity={inspect['best_unit_selectivity']:.3f} "
          f"in {elapsed:.1f}s")

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "canonical_model.npz"),
                        W_th=model.W_th, W_hv=model.W_hv,
                        b_top=model.b_top, b_h=model.b_h, b_v=model.b_v,
                        R_vh=model.R_vh, R_ht=model.R_ht,
                        c_h=model.c_h, c_top=model.c_top,
                        seed=np.array([seed]),
                        n_passes=np.array([n_passes]),
                        n_hidden=np.array([n_hidden]),
                        n_top=np.array([n_top]),
                        lr=np.array([lr]),
                        batch_size=np.array([batch_size]),
                        p_on=np.array([p_on]),
                        elapsed_sec=np.array([elapsed]),
                        final_nll_bits=np.array([final_nll]),
                        final_dir_acc=np.array([final_dir_acc]),
                        best_unit=np.array([inspect["best_unit"]]),
                        best_unit_selectivity=np.array(
                            [inspect["best_unit_selectivity"]]),
                        )
    np.savez_compressed(os.path.join(outdir, "canonical_history.npz"),
                        step=np.array(history["step"]),
                        samples=np.array(history["samples"]),
                        is_nll_bits=np.array(history["is_nll_bits"]),
                        dir_acc=np.array(history["dir_acc"]))
    np.savez_compressed(os.path.join(outdir, "canonical_snapshots.npz"),
                        steps=np.array([s[0] for s in snapshots]),
                        W_hv=np.array([s[1] for s in snapshots]),
                        W_th=np.array([s[2] for s in snapshots]),
                        b_v=np.array([s[3] for s in snapshots]),
                        b_h=np.array([s[4] for s in snapshots]),
                        b_top=np.array([s[5] for s in snapshots]),
                        fantasy_samples=np.array([s[6] for s in snapshots]),
                        is_nll_bits=np.array([s[7] for s in snapshots]),
                        dir_acc=np.array([s[8] for s in snapshots]))
    return model, history, snapshots


def load_model(path_dir: str) -> hs.HelmholtzMachine:
    a = np.load(os.path.join(path_dir, "canonical_model.npz"))
    m = hs.HelmholtzMachine(n_hidden=int(a["n_hidden"][0]),
                            n_top=int(a["n_top"][0]),
                            p_on=float(a["p_on"][0]),
                            seed=int(a["seed"][0]))
    m.W_th = a["W_th"]
    m.W_hv = a["W_hv"]
    m.b_top = a["b_top"]
    m.b_h = a["b_h"]
    m.b_v = a["b_v"]
    m.R_vh = a["R_vh"]
    m.R_ht = a["R_ht"]
    m.c_h = a["c_h"]
    m.c_top = a["c_top"]
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-passes", type=int, default=1_500_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-hidden", type=int, default=hs.N_HID_DEFAULT)
    p.add_argument("--n-top", type=int, default=hs.N_TOP_DEFAULT)
    p.add_argument("--p-on", type=float, default=hs.P_ON)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--snapshot-every", type=int, default=50_000)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()
    train_and_save(args.seed, args.n_passes, args.lr, args.batch_size,
                   args.eval_every, args.snapshot_every, args.outdir,
                   n_hidden=args.n_hidden, n_top=args.n_top, p_on=args.p_on)


if __name__ == "__main__":
    main()
