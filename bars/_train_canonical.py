"""Run the canonical 2M-sample training and save the trained model + history.

Saved artifacts (in `viz/`):
    canonical_model.npz   -- weights + biases of the trained Helmholtz machine
    canonical_history.npz -- KL trajectory + step counts

This script is invoked by `make_bars_gif.py` and `visualize_bars.py` so that
both share the same trained network rather than re-training twice.
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

import bars


def train_and_save(seed: int, n_steps: int, lr: float, batch_size: int,
                   eval_every: int, snapshot_every: int, outdir: str
                   ) -> tuple[bars.HelmholtzMachine, dict, list]:
    rng = np.random.default_rng(seed)
    model = bars.HelmholtzMachine(seed=seed)
    snapshots = []  # (step, W_hv copy, samples-from-model)

    def cb(step, m, history):
        if step % snapshot_every == 0 or step == n_steps:
            v_fant, _, _ = m.generate(64)
            snapshots.append((step,
                              m.W_hv.copy(),
                              m.W_th.copy(),
                              m.b_v.copy(),
                              m.b_h.copy(),
                              m.b_top.copy(),
                              v_fant.copy(),
                              float(history["kl_bits"][-1])))
            print(f"  snapshot at step {step:>10d}  KL={history['kl_bits'][-1]:.4f}")

    t0 = time.time()
    history = bars.wake_sleep(model, rng, n_steps=n_steps, lr=lr,
                              batch_size=batch_size,
                              eval_every=eval_every,
                              eval_callback=cb)
    elapsed = time.time() - t0
    final_kl = bars.asymmetric_kl(model)
    print(f"Done. Final KL={final_kl:.4f} bits in {elapsed:.1f}s")

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "canonical_model.npz"),
                        W_th=model.W_th, W_hv=model.W_hv,
                        b_top=model.b_top, b_h=model.b_h, b_v=model.b_v,
                        R_vh=model.R_vh, R_ht=model.R_ht,
                        c_h=model.c_h, c_top=model.c_top,
                        seed=np.array([seed]),
                        n_steps=np.array([n_steps]),
                        lr=np.array([lr]),
                        batch_size=np.array([batch_size]),
                        elapsed_sec=np.array([elapsed]),
                        final_kl_bits=np.array([final_kl]))
    np.savez_compressed(os.path.join(outdir, "canonical_history.npz"),
                        step=np.array(history["step"]),
                        samples=np.array(history["samples"]),
                        kl_bits=np.array(history["kl_bits"]),
                        neg_log_lik=np.array(history["neg_log_lik"]))
    # snapshots: pickle as object array
    np.savez_compressed(os.path.join(outdir, "canonical_snapshots.npz"),
                        steps=np.array([s[0] for s in snapshots]),
                        W_hv=np.array([s[1] for s in snapshots]),
                        W_th=np.array([s[2] for s in snapshots]),
                        b_v=np.array([s[3] for s in snapshots]),
                        b_h=np.array([s[4] for s in snapshots]),
                        b_top=np.array([s[5] for s in snapshots]),
                        fantasy_samples=np.array([s[6] for s in snapshots]),
                        kl_bits=np.array([s[7] for s in snapshots]))
    return model, history, snapshots


def load_model(path_dir: str) -> bars.HelmholtzMachine:
    a = np.load(os.path.join(path_dir, "canonical_model.npz"))
    m = bars.HelmholtzMachine(n_hidden=int(a["W_hv"].shape[0]),
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
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=20_000)
    p.add_argument("--snapshot-every", type=int, default=20_000)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()
    train_and_save(args.seed, args.n_steps, args.lr, args.batch_size,
                   args.eval_every, args.snapshot_every, args.outdir)


if __name__ == "__main__":
    main()
