"""
Constellations -- 2D point-cloud part-whole grouping.

Numpy reproduction of Kosiorek, Sabour, Teh & Hinton, "Stacked capsule
autoencoders" (NeurIPS 2019), constellations experiment.

Each example is the union of K=3 affine-transformed copies of fixed point
templates (square, triangle-with-extra, triangle = 4+4+3 = 11 points). The
network sees the 11 points in random order, must figure out which point
belongs to which template, and recover the transforms.

Architecture:
  Encoder = set transformer (per-point embedding + self-attention + pooling-
            by-multihead-attention with K=3 learned seed queries).
  Decoder = K capsules. Each capsule maps its embedding to a similarity
            transform (scale, rotation, tx, ty) that gets applied to its
            assigned hardcoded template, producing predicted points.

Loss = symmetric Chamfer distance between input points and decoded points
       (the cleanest differentiable analogue of the paper's Gaussian-mixture
       part likelihood; see deviation #2 in the README).

Permutation invariance: the encoder uses scaled dot-product attention only,
so the K capsule embeddings are invariant to the input point order.

Required by spec.py:
  affine_transform, generate_constellation, build_set_transformer_encoder,
  build_capsule_decoder, train, part_capsule_recovery_accuracy.
"""
from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Templates (per spec)
# ----------------------------------------------------------------------
#
# 4 + 4 + 3 = 11 points per example. We center each template on its
# centroid so the random translation is the only "where" parameter.

_RAW_TEMPLATES = (
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),   # square
    np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.0]]),   # triangle-with-extra
    np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]),             # triangle
)
TEMPLATES = tuple((t - t.mean(axis=0)).astype(np.float32)
                  for t in _RAW_TEMPLATES)
TEMPLATE_SIZES = tuple(t.shape[0] for t in TEMPLATES)             # (4, 4, 3)
N_TEMPLATES = len(TEMPLATES)                                       # 3
N_POINTS = sum(TEMPLATE_SIZES)                                     # 11


# ----------------------------------------------------------------------
# Affine transform
# ----------------------------------------------------------------------
#
# Similarity transform: scale * R(theta) @ p + (tx, ty). The decoder
# outputs the same parameterisation (4 scalars per capsule).

def affine_transform(points: np.ndarray,
                     scale: float | None = None,
                     theta: float | None = None,
                     tx: float | None = None,
                     ty: float | None = None,
                     scale_range: tuple[float, float] = (0.5, 1.5),
                     trans_range: float = 3.0,
                     rng: np.random.Generator | None = None,
                     ) -> tuple[np.ndarray, dict]:
    """Apply a random (or specified) similarity transform to `points`.

    Returns (transformed_points, params) where params is a dict with the
    keys 'scale', 'theta', 'tx', 'ty'.
    """
    rng = rng or np.random.default_rng()
    if scale is None:
        scale = float(rng.uniform(*scale_range))
    if theta is None:
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
    if tx is None:
        tx = float(rng.uniform(-trans_range, trans_range))
    if ty is None:
        ty = float(rng.uniform(-trans_range, trans_range))
    c, s = np.cos(theta), np.sin(theta)
    R = scale * np.array([[c, -s], [s, c]], dtype=np.float32)
    out = (points @ R.T) + np.array([tx, ty], dtype=np.float32)
    return out.astype(np.float32), dict(scale=scale, theta=theta, tx=tx, ty=ty)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

def generate_constellation(rng: np.random.Generator | None = None
                           ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """One example: union of K=3 affine-transformed templates, then shuffled.

    Returns:
      points: (11, 2) shuffled coordinates
      labels: (11,)  template index in {0, 1, 2} per point
      transforms: list of 3 dicts (the ground-truth similarity params per template)
    """
    rng = rng or np.random.default_rng()
    pts_list = []
    lbl_list = []
    transforms = []
    for k, tmpl in enumerate(TEMPLATES):
        tx_pts, params = affine_transform(tmpl, rng=rng)
        pts_list.append(tx_pts)
        lbl_list.append(np.full(tmpl.shape[0], k, dtype=np.int32))
        transforms.append(params)
    points = np.concatenate(pts_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    perm = rng.permutation(points.shape[0])
    return points[perm].astype(np.float32), labels[perm], transforms


def make_dataset(n_examples: int, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Stack many constellations. Returns (points (n, 11, 2), labels (n, 11))."""
    pts = np.empty((n_examples, N_POINTS, 2), dtype=np.float32)
    lbl = np.empty((n_examples, N_POINTS), dtype=np.int32)
    for i in range(n_examples):
        p, l, _ = generate_constellation(rng)
        pts[i] = p
        lbl[i] = l
    return pts, lbl


# ----------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ----------------------------------------------------------------------
# Model -- set-transformer encoder + capsule decoder
# ----------------------------------------------------------------------
#
# Convention: B = batch size, N = 11 = total points per example,
#             K = 3 = number of object capsules / templates,
#             D = embedding dim, T_ATTN = single-head attention.

class ConstellationsModel:
    """Permutation-invariant point-set encoder + capsule decoder.

    Encoder pipeline (forward):
      x (B, N, 2)
        -> point-wise linear+ReLU      (B, N, D)
        -> self-attention (single SAB) (B, N, D)
        -> point-wise FFN              (B, N, D)
        -> PMA: K learned seeds cross-attend over points  -> (B, K, D)
        -> per-capsule head            -> (B, K, 4) = (log_scale, theta, tx, ty)

    The K seed queries are LEARNED parameters; this is what breaks the
    K-fold symmetry and lets each capsule lock onto a specific template.

    Decoder:
      capsule k applies its similarity transform to TEMPLATES[k] to produce
      decoded points.

    Loss:
      symmetric chamfer between input cloud and decoded cloud.
    """

    def __init__(self,
                 d_embed: int = 32,
                 d_ffn: int = 64,
                 n_object_capsules: int = N_TEMPLATES,
                 init_scale: float = 0.5,
                 seed: int = 0):
        if n_object_capsules != N_TEMPLATES:
            raise ValueError(
                f"K is fixed to {N_TEMPLATES} (one capsule per hardcoded "
                f"template). Got n_object_capsules={n_object_capsules}.")
        self.D = d_embed
        self.F = d_ffn
        self.K = n_object_capsules
        self.rng = np.random.default_rng(seed)

        def he(shape, fan_in):
            return (init_scale * self.rng.standard_normal(shape)
                    * np.sqrt(2.0 / fan_in)).astype(np.float32)

        D = self.D
        # Per-point embedding 2 -> D (with a small relu MLP)
        self.W_in = he((2, D), 2)
        self.b_in = np.zeros(D, dtype=np.float32)

        # Self-attention block (SAB): single-head, dim D
        self.W_q = he((D, D), D)
        self.W_k = he((D, D), D)
        self.W_v = he((D, D), D)
        self.W_o = he((D, D), D)

        # Position-wise FFN after SAB
        self.W_f1 = he((D, self.F), D)
        self.b_f1 = np.zeros(self.F, dtype=np.float32)
        self.W_f2 = he((self.F, D), self.F)
        self.b_f2 = np.zeros(D, dtype=np.float32)

        # PMA: K learned seed queries, single-head cross-attention
        self.S = he((self.K, D), D)            # (K, D) seeds
        self.W_qp = he((D, D), D)
        self.W_kp = he((D, D), D)
        self.W_vp = he((D, D), D)
        self.W_op = he((D, D), D)

        # Per-capsule head: D -> 4 (log_scale, theta, tx, ty)
        self.W_dec = he((D, 4), D)
        self.b_dec = np.zeros(4, dtype=np.float32)

    @property
    def param_names(self):
        return ("W_in", "b_in",
                "W_q", "W_k", "W_v", "W_o",
                "W_f1", "b_f1", "W_f2", "b_f2",
                "S", "W_qp", "W_kp", "W_vp", "W_op",
                "W_dec", "b_dec")

    def zero_like_params(self):
        return {n: np.zeros_like(getattr(self, n)) for n in self.param_names}

    # -- forward -----------------------------------------------------------

    def forward(self, x: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        x: (B, N, 2) float32 input cloud.
        Returns (decoded_points, point_capsule, capsule_params, cache).
          decoded_points: (B, M_total=11, 2) -- concatenated, in
                          [template_0_pts, template_1_pts, template_2_pts] order.
          point_capsule: (M_total,) int -- which capsule each decoded point came from.
          capsule_params: (B, K, 4) -- (log_scale, theta, tx, ty) per capsule.
        """
        B, N, _ = x.shape
        D = self.D

        # 1) Per-point embedding 2 -> D (linear; no ReLU here so signed
        # coordinates aren't gated -- ReLU comes later in the FFN).
        h_in = x @ self.W_in + self.b_in                          # (B, N, D)

        # 2) Self-attention block (SAB, single-head, no LayerNorm).
        Q = h_in @ self.W_q                                       # (B, N, D)
        Kk = h_in @ self.W_k
        V = h_in @ self.W_v
        scale = 1.0 / np.sqrt(D)
        scores = np.matmul(Q, Kk.transpose(0, 2, 1)) * scale      # (B, N, N)
        attn = softmax(scores, axis=-1)
        ctx = np.matmul(attn, V)                                  # (B, N, D)
        sab_out = ctx @ self.W_o                                  # (B, N, D)
        h_sab = h_in + sab_out                                    # residual

        # 3) Position-wise FFN (Linear -> ReLU -> Linear, residual).
        f1_pre = h_sab @ self.W_f1 + self.b_f1                    # (B, N, F)
        f1 = relu(f1_pre)
        ffn_out = f1 @ self.W_f2 + self.b_f2                      # (B, N, D)
        h_enc = h_sab + ffn_out                                   # (B, N, D)

        # 4) PMA: K learned seeds cross-attend over h_enc (single-head).
        S = self.S                                                 # (K, D)
        Qp = S @ self.W_qp                                         # (K, D)
        Kp = h_enc @ self.W_kp                                     # (B, N, D)
        Vp = h_enc @ self.W_vp                                     # (B, N, D)
        # Broadcast Qp to (B, K, D) for matmul with Kp^T -> (B, K, N).
        Qp_b = np.broadcast_to(Qp[None, :, :], (B, self.K, D))
        scores_p = np.matmul(Qp_b, Kp.transpose(0, 2, 1)) * scale  # (B, K, N)
        attn_p = softmax(scores_p, axis=-1)
        ctx_p = np.matmul(attn_p, Vp)                              # (B, K, D)
        caps = ctx_p @ self.W_op                                   # (B, K, D)

        # 5) Per-capsule decode head: D -> 4 (log_scale, theta, tx, ty)
        params = caps @ self.W_dec + self.b_dec                    # (B, K, 4)

        # 6) Decode: apply each capsule's similarity to its template.
        decoded, point_capsule = self._decode(params)

        cache = dict(x=x, h_in=h_in, Q=Q, Kk=Kk, V=V, attn=attn, ctx=ctx,
                     sab_out=sab_out, h_sab=h_sab, f1_pre=f1_pre, f1=f1,
                     ffn_out=ffn_out, h_enc=h_enc,
                     Qp=Qp, Qp_b=Qp_b, Kp=Kp, Vp=Vp, attn_p=attn_p,
                     ctx_p=ctx_p, caps=caps, params=params,
                     decoded=decoded, point_capsule=point_capsule)
        return decoded, point_capsule, params, cache

    def _decode(self, params: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """params: (B, K, 4) -> decoded points (B, M_total, 2) and capsule index per point."""
        B, K, _ = params.shape
        out_chunks = []
        cap_idx = []
        for k in range(K):
            log_s = params[:, k, 0]            # (B,)
            theta = params[:, k, 1]
            tx = params[:, k, 2]
            ty = params[:, k, 3]
            s = np.exp(log_s)                  # (B,)
            c = np.cos(theta)
            sn = np.sin(theta)
            # R[k] applied to TEMPLATES[k]: (M_k, 2)
            tmpl = TEMPLATES[k]                # (M_k, 2)
            # rotated = tmpl @ R^T = tmpl @ [[c, s],[-s, c]]
            # for batched: out[b, n, :] = s[b] * (c[b]*tmpl[n,0] - sn[b]*tmpl[n,1] + ..., ...)
            rotated = np.empty((B, tmpl.shape[0], 2), dtype=np.float32)
            rotated[:, :, 0] = (c[:, None] * tmpl[None, :, 0]
                                - sn[:, None] * tmpl[None, :, 1])
            rotated[:, :, 1] = (sn[:, None] * tmpl[None, :, 0]
                                + c[:, None] * tmpl[None, :, 1])
            scaled = s[:, None, None] * rotated
            translated = scaled + np.stack([tx, ty], axis=-1)[:, None, :]
            out_chunks.append(translated)
            cap_idx.append(np.full(tmpl.shape[0], k, dtype=np.int32))
        decoded = np.concatenate(out_chunks, axis=1)               # (B, M_total, 2)
        point_capsule = np.concatenate(cap_idx, axis=0)            # (M_total,)
        return decoded.astype(np.float32), point_capsule

    # -- chamfer loss + backward through decode ---------------------------

    def chamfer_loss_and_dparams(self, decoded: np.ndarray, x: np.ndarray
                                  ) -> tuple[float, np.ndarray]:
        """Symmetric chamfer between decoded (B, M, 2) and input x (B, N, 2).

        Returns (loss, d_decoded), where d_decoded has shape (B, M, 2).
        For chamfer with hard argmin, the gradient w.r.t. each point is
        the residual to its nearest neighbor in the other set (twice, once
        from each direction).
        """
        B, M, _ = decoded.shape
        N = x.shape[1]
        # pairwise squared distances (B, M, N)
        diff = decoded[:, :, None, :] - x[:, None, :, :]
        sq = (diff ** 2).sum(axis=-1)                              # (B, M, N)

        # For each decoded point, nearest input
        idx_d2x = np.argmin(sq, axis=2)                            # (B, M)
        # For each input point, nearest decoded
        idx_x2d = np.argmin(sq, axis=1)                            # (B, N)

        b_idx = np.arange(B)[:, None]
        m_idx = np.arange(M)[None, :]
        n_idx = np.arange(N)[None, :]
        nn_for_d = x[b_idx, idx_d2x]                               # (B, M, 2)
        nn_for_x = decoded[b_idx, idx_x2d]                         # (B, N, 2)

        # Chamfer loss: average per-point squared distance, in both directions.
        loss_d = ((decoded - nn_for_d) ** 2).sum(axis=-1).mean()
        loss_x = ((x - nn_for_x) ** 2).sum(axis=-1).mean()
        loss = float(loss_d + loss_x)

        # Gradient through the d->x term: 2/(B*M) * (decoded - nn_for_d)
        d_decoded = (2.0 / (B * M)) * (decoded - nn_for_d)         # (B, M, 2)

        # Gradient through the x->d term: scatter -2/(B*N)*(x - nn_for_x)
        # back onto the decoded points that were "winners" for each input.
        contrib = (2.0 / (B * N)) * (nn_for_x - x)                 # (B, N, 2)
        # Add to decoded[b, idx_x2d[b, n]] the row contrib[b, n].
        np.add.at(d_decoded, (b_idx, idx_x2d), contrib)
        return loss, d_decoded.astype(np.float32)

    # -- backward through decoder ------------------------------------------

    def backward(self, cache: dict, d_decoded: np.ndarray) -> dict:
        """Backprop d_decoded (B, M_total, 2) through the whole network.

        Returns grads dict with one entry per param_name.
        """
        B = d_decoded.shape[0]
        D = self.D
        K = self.K

        # ---- 6) decode: similarity transform -----------------------------
        params = cache["params"]                  # (B, K, 4)
        d_params = np.zeros_like(params)          # (B, K, 4)

        # Each capsule's chunk of d_decoded
        cur = 0
        for k in range(K):
            tmpl = TEMPLATES[k]
            M_k = tmpl.shape[0]
            d_dec_k = d_decoded[:, cur:cur + M_k, :]              # (B, M_k, 2)
            cur += M_k

            log_s = params[:, k, 0]
            theta = params[:, k, 1]
            s = np.exp(log_s)
            c = np.cos(theta)
            sn = np.sin(theta)

            # decoded_k[b,n,0] = s[b] * (c[b]*tmpl[n,0] - sn[b]*tmpl[n,1]) + tx[b]
            # decoded_k[b,n,1] = s[b] * (sn[b]*tmpl[n,0] + c[b]*tmpl[n,1]) + ty[b]
            t0 = tmpl[None, :, 0]                                  # (1, M_k)
            t1 = tmpl[None, :, 1]
            r0 = c[:, None] * t0 - sn[:, None] * t1                # (B, M_k)
            r1 = sn[:, None] * t0 + c[:, None] * t1

            d_x = d_dec_k[:, :, 0]                                 # (B, M_k)
            d_y = d_dec_k[:, :, 1]

            # d/d log_s : decoded depends on s = exp(log_s), so
            #   d/d log_s = s * (d_x*r0 + d_y*r1).sum_n
            d_log_s = (s * ((d_x * r0).sum(axis=1)
                            + (d_y * r1).sum(axis=1)))             # (B,)

            # d/d theta : derivative of rotated point w.r.t. theta:
            #   d r0 / d theta = -sn*t0 - c*t1   = -r1
            #   d r1 / d theta =  c*t0 - sn*t1   =  r0
            d_theta = (s * ((d_x * (-r1)).sum(axis=1)
                            + (d_y * r0).sum(axis=1)))             # (B,)

            d_tx = d_x.sum(axis=1)                                 # (B,)
            d_ty = d_y.sum(axis=1)                                 # (B,)

            d_params[:, k, 0] = d_log_s
            d_params[:, k, 1] = d_theta
            d_params[:, k, 2] = d_tx
            d_params[:, k, 3] = d_ty

        # ---- 5) per-capsule head : D -> 4 --------------------------------
        # params = caps @ W_dec + b_dec
        caps = cache["caps"]                                       # (B, K, D)
        # d_W_dec = sum_b sum_k caps[b,k,:].T @ d_params[b,k,:]
        d_W_dec = np.einsum("bki,bkj->ij", caps, d_params)         # (D, 4)
        d_b_dec = d_params.sum(axis=(0, 1))                        # (4,)
        d_caps = d_params @ self.W_dec.T                           # (B, K, D)

        # ---- 4) PMA cross-attention --------------------------------------
        # caps = ctx_p @ W_op
        ctx_p = cache["ctx_p"]
        d_W_op = np.einsum("bkd,bke->de", ctx_p, d_caps)            # (D, D)
        d_ctx_p = d_caps @ self.W_op.T                              # (B, K, D)

        # ctx_p = attn_p @ Vp   ; attn_p (B, K, N), Vp (B, N, D)
        attn_p = cache["attn_p"]
        Vp = cache["Vp"]
        d_attn_p = np.matmul(d_ctx_p, Vp.transpose(0, 2, 1))        # (B, K, N)
        d_Vp = np.matmul(attn_p.transpose(0, 2, 1), d_ctx_p)        # (B, N, D)

        # softmax backprop
        # attn_p = softmax(scores_p)
        # d scores_p = attn_p * (d_attn_p - sum_j attn_p[j] d_attn_p[j])
        d_scores_p = attn_p * (
            d_attn_p - (attn_p * d_attn_p).sum(axis=-1, keepdims=True))
        scale = 1.0 / np.sqrt(D)
        d_scores_p = d_scores_p * scale

        # scores_p = Qp_b @ Kp.T ; Qp_b (B, K, D), Kp (B, N, D)
        Qp_b = cache["Qp_b"]
        Kp = cache["Kp"]
        d_Qp_b = np.matmul(d_scores_p, Kp)                          # (B, K, D)
        d_Kp = np.matmul(d_scores_p.transpose(0, 2, 1), Qp_b)       # (B, N, D)

        # Qp_b is broadcast from Qp = S @ W_qp
        d_Qp = d_Qp_b.sum(axis=0)                                   # (K, D)

        # Qp = S @ W_qp  ;  S is (K, D), W_qp is (D, D)
        d_S_pma = d_Qp @ self.W_qp.T                                # (K, D)
        d_W_qp = self.S.T @ d_Qp                                     # (D, D)

        # Kp = h_enc @ W_kp
        h_enc = cache["h_enc"]
        d_W_kp = np.einsum("bni,bnj->ij", h_enc, d_Kp)              # (D, D)
        d_h_enc_kp = d_Kp @ self.W_kp.T                              # (B, N, D)

        # Vp = h_enc @ W_vp
        d_W_vp = np.einsum("bni,bnj->ij", h_enc, d_Vp)              # (D, D)
        d_h_enc_vp = d_Vp @ self.W_vp.T                              # (B, N, D)

        d_h_enc = d_h_enc_kp + d_h_enc_vp                            # (B, N, D)

        # ---- 3) FFN residual : h_enc = h_sab + ffn_out -------------------
        d_h_sab = d_h_enc.copy()                                     # residual path
        d_ffn_out = d_h_enc                                           # to FFN

        # ffn_out = f1 @ W_f2 + b_f2
        f1 = cache["f1"]
        d_W_f2 = np.einsum("bni,bnj->ij", f1, d_ffn_out)
        d_b_f2 = d_ffn_out.sum(axis=(0, 1))
        d_f1 = d_ffn_out @ self.W_f2.T                               # (B, N, F)

        # f1 = relu(f1_pre)
        f1_pre = cache["f1_pre"]
        d_f1_pre = d_f1 * (f1_pre > 0).astype(np.float32)

        # f1_pre = h_sab @ W_f1 + b_f1
        h_sab = cache["h_sab"]
        d_W_f1 = np.einsum("bni,bnj->ij", h_sab, d_f1_pre)
        d_b_f1 = d_f1_pre.sum(axis=(0, 1))
        d_h_sab += d_f1_pre @ self.W_f1.T                            # (B, N, D)

        # ---- 2) SAB : h_sab = h_in + sab_out -----------------------------
        d_h_in = d_h_sab.copy()
        d_sab_out = d_h_sab                                           # to SAB

        # sab_out = ctx @ W_o
        ctx = cache["ctx"]
        d_W_o = np.einsum("bni,bnj->ij", ctx, d_sab_out)
        d_ctx = d_sab_out @ self.W_o.T                                # (B, N, D)

        # ctx = attn @ V
        attn = cache["attn"]
        V = cache["V"]
        d_attn = np.matmul(d_ctx, V.transpose(0, 2, 1))               # (B, N, N)
        d_V = np.matmul(attn.transpose(0, 2, 1), d_ctx)               # (B, N, D)

        # softmax(scores) -> attn ; same identity as before
        d_scores = attn * (d_attn - (attn * d_attn).sum(axis=-1, keepdims=True))
        d_scores = d_scores * scale

        # scores = Q @ K^T
        Q = cache["Q"]
        Kk = cache["Kk"]
        d_Q = np.matmul(d_scores, Kk)                                  # (B, N, D)
        d_K = np.matmul(d_scores.transpose(0, 2, 1), Q)                # (B, N, D)

        # Q = h_in @ W_q ; K = h_in @ W_k ; V = h_in @ W_v
        h_in = cache["h_in"]
        d_W_q = np.einsum("bni,bnj->ij", h_in, d_Q)
        d_W_k = np.einsum("bni,bnj->ij", h_in, d_K)
        d_W_v = np.einsum("bni,bnj->ij", h_in, d_V)
        d_h_in += d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T

        # ---- 1) input embedding ------------------------------------------
        # h_in = x @ W_in + b_in
        x = cache["x"]
        d_W_in = np.einsum("bni,bnj->ij", x, d_h_in)                   # (2, D)
        d_b_in = d_h_in.sum(axis=(0, 1))

        return dict(W_in=d_W_in, b_in=d_b_in,
                    W_q=d_W_q, W_k=d_W_k, W_v=d_W_v, W_o=d_W_o,
                    W_f1=d_W_f1, b_f1=d_b_f1, W_f2=d_W_f2, b_f2=d_b_f2,
                    S=d_S_pma, W_qp=d_W_qp, W_kp=d_W_kp, W_vp=d_W_vp,
                    W_op=d_W_op, W_dec=d_W_dec, b_dec=d_b_dec)


# ----------------------------------------------------------------------
# Recovery accuracy
# ----------------------------------------------------------------------

def _all_permutations(K: int) -> list[tuple[int, ...]]:
    """Enumerate all K! permutations as tuples (small K, brute-force is fine)."""
    if K == 1:
        return [(0,)]
    out = []
    for k in range(K):
        rest = [j for j in range(K) if j != k]
        for sub in _all_permutations(K - 1):
            out.append((k,) + tuple(rest[i] for i in sub))
    return out


def part_capsule_recovery_accuracy(model: ConstellationsModel,
                                   data: tuple[np.ndarray, np.ndarray],
                                   permutation_invariant: bool = True
                                   ) -> float:
    """Per-point recovery accuracy on a held-out (points, labels) set.

    For each input point, the predicted template is the capsule index of its
    nearest decoded point.

    permutation_invariant=True (default): pick the best K!-way assignment
        of capsules to ground-truth template indices on a *per-example*
        basis, and report the per-point hit rate under that match. The
        model's K=3 capsule order is arbitrary -- the cluster identities
        are recovered up to a permutation of the K labels -- so reporting
        the raw match would conflate "the model failed to cluster the
        points" with "the model clustered them but capsule 1 happens to
        decode template 2's shape." The 3! = 6 permutations are
        enumerated explicitly.

    permutation_invariant=False: report the strict capsule-id == label rate.
    """
    points, labels = data
    decoded, point_capsule, _, _ = model.forward(points)
    # squared distances (B, N, M)
    diff = points[:, :, None, :] - decoded[:, None, :, :]
    sq = (diff ** 2).sum(axis=-1)
    nn = np.argmin(sq, axis=2)                          # (B, N)
    pred_caps = point_capsule[nn]                       # (B, N)

    if not permutation_invariant:
        return float((pred_caps == labels).mean())

    K = model.K
    perms = _all_permutations(K)
    # For each example, evaluate every permutation and take the max.
    B = pred_caps.shape[0]
    best = np.zeros(B, dtype=np.float32)
    for perm in perms:
        perm_arr = np.array(perm, dtype=np.int32)
        relabelled = perm_arr[pred_caps]                 # (B, N)
        score = (relabelled == labels).mean(axis=1)      # (B,)
        best = np.maximum(best, score)
    return float(best.mean())


# ----------------------------------------------------------------------
# Convenience public constructors (per spec)
# ----------------------------------------------------------------------

def build_set_transformer_encoder(d_embed: int = 32, d_ffn: int = 64,
                                  n_object_capsules: int = N_TEMPLATES,
                                  seed: int = 0):
    """Return the `ConstellationsModel` (encoder + decoder live in one class)."""
    return ConstellationsModel(d_embed=d_embed, d_ffn=d_ffn,
                               n_object_capsules=n_object_capsules,
                               seed=seed)


def build_capsule_decoder(n_object_capsules: int = N_TEMPLATES,
                          template_set=TEMPLATES):
    """The capsule decoder is `model._decode(params)` -- it is a pure function
    of the capsule-emitted similarity parameters and the hardcoded templates,
    so we expose a tiny closure.
    """
    if template_set is not TEMPLATES:
        raise ValueError("This implementation hardcodes TEMPLATES; "
                         "swap the module-level constant to change them.")

    def decode(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Stand-alone version that matches ConstellationsModel._decode()
        B, K, _ = params.shape
        out_chunks = []
        cap_idx = []
        for k in range(K):
            log_s = params[:, k, 0]
            theta = params[:, k, 1]
            tx = params[:, k, 2]
            ty = params[:, k, 3]
            s = np.exp(log_s)
            c = np.cos(theta)
            sn = np.sin(theta)
            tmpl = template_set[k]
            rotated = np.empty((B, tmpl.shape[0], 2), dtype=np.float32)
            rotated[:, :, 0] = (c[:, None] * tmpl[None, :, 0]
                                - sn[:, None] * tmpl[None, :, 1])
            rotated[:, :, 1] = (sn[:, None] * tmpl[None, :, 0]
                                + c[:, None] * tmpl[None, :, 1])
            scaled = s[:, None, None] * rotated
            translated = scaled + np.stack([tx, ty], axis=-1)[:, None, :]
            out_chunks.append(translated)
            cap_idx.append(np.full(tmpl.shape[0], k, dtype=np.int32))
        return (np.concatenate(out_chunks, axis=1).astype(np.float32),
                np.concatenate(cap_idx, axis=0))

    return decode


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(model: ConstellationsModel | None = None,
          n_epochs: int = 30,
          steps_per_epoch: int = 200,
          batch_size: int = 32,
          lr: float = 3e-3,
          beta1: float = 0.9,
          beta2: float = 0.999,
          eps: float = 1e-8,
          n_object_capsules: int = N_TEMPLATES,
          d_embed: int = 32,
          d_ffn: int = 64,
          val_size: int = 256,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 100,
          verbose: bool = True
          ) -> tuple[ConstellationsModel, dict]:
    """Train the constellations model with Adam.

    Returns (model, history). history keys: step, epoch, loss, val_loss,
    val_recovery, snapshots-of-{x, decoded, point_capsule, labels} on the
    validation set every snapshot_every steps via `snapshot_callback`.
    """
    rng = np.random.default_rng(seed)
    if model is None:
        model = ConstellationsModel(d_embed=d_embed, d_ffn=d_ffn,
                                    n_object_capsules=n_object_capsules,
                                    seed=seed)

    # Adam state
    adam_m = model.zero_like_params()
    adam_v = model.zero_like_params()
    adam_t = 0

    val_rng = np.random.default_rng(seed + 1)
    val_x, val_y = make_dataset(val_size, val_rng)

    history = {"step": [], "epoch": [], "loss": [],
               "val_loss": [], "val_recovery": []}

    if verbose:
        n_params = sum(int(np.prod(getattr(model, n).shape))
                       for n in model.param_names)
        print(f"# constellations: K={model.K} capsules, D={model.D}, "
              f"F={model.F}, params={n_params}")
        print(f"# train: {steps_per_epoch} steps/epoch x {n_epochs} epochs "
              f"x batch {batch_size} (lr={lr})")
        print(f"# val: {val_size} examples")

    step = 0
    t_start = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for s in range(steps_per_epoch):
            x_batch, _ = make_dataset(batch_size, rng)
            decoded, _, _, cache = model.forward(x_batch)
            loss, d_decoded = model.chamfer_loss_and_dparams(decoded, x_batch)
            grads = model.backward(cache, d_decoded)

            adam_t += 1
            bc1 = 1.0 - beta1 ** adam_t
            bc2 = 1.0 - beta2 ** adam_t
            for k_, g in grads.items():
                m_buf = adam_m[k_]
                v_buf = adam_v[k_]
                m_buf[...] = beta1 * m_buf + (1.0 - beta1) * g
                v_buf[...] = beta2 * v_buf + (1.0 - beta2) * (g * g)
                update = lr * (m_buf / bc1) / (np.sqrt(v_buf / bc2) + eps)
                getattr(model, k_)[...] -= update
            epoch_loss += loss
            step += 1

            if snapshot_callback is not None and (step % snapshot_every == 0):
                snapshot_callback(step, model, history, val_x, val_y)

        # End-of-epoch validation
        val_dec, _, _, _ = model.forward(val_x)
        val_loss, _ = model.chamfer_loss_and_dparams(val_dec, val_x)
        val_acc = part_capsule_recovery_accuracy(model, (val_x, val_y))
        history["step"].append(step)
        history["epoch"].append(epoch + 1)
        history["loss"].append(epoch_loss / steps_per_epoch)
        history["val_loss"].append(val_loss)
        history["val_recovery"].append(val_acc)

        if verbose:
            elapsed = time.time() - t_start
            print(f"epoch {epoch+1:2d}/{n_epochs}  "
                  f"train_chamfer={epoch_loss/steps_per_epoch:.4f}  "
                  f"val_chamfer={val_loss:.4f}  "
                  f"recovery={val_acc*100:.1f}%  ({elapsed:.1f}s)",
                  flush=True)

    return model, history


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-templates", type=int, default=N_TEMPLATES,
                   help=f"K (fixed at {N_TEMPLATES} per spec)")
    p.add_argument("--n-object-capsules", type=int, default=N_TEMPLATES)
    p.add_argument("--d-embed", type=int, default=32)
    p.add_argument("--d-ffn", type=int, default=64)
    p.add_argument("--val-size", type=int, default=256)
    args = p.parse_args()

    if args.n_templates != N_TEMPLATES:
        raise ValueError(f"K is fixed at {N_TEMPLATES}; got --n-templates="
                         f"{args.n_templates}")

    model, history = train(
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        n_object_capsules=args.n_object_capsules,
        d_embed=args.d_embed,
        d_ffn=args.d_ffn,
        val_size=args.val_size,
        seed=args.seed,
    )

    print(f"\nFinal train chamfer: {history['loss'][-1]:.4f}")
    print(f"Final val chamfer:   {history['val_loss'][-1]:.4f}")
    print(f"Final part-capsule recovery: "
          f"{history['val_recovery'][-1] * 100:.1f}%")


if __name__ == "__main__":
    main()
