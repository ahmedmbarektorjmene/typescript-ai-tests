import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton for High Performance Attention
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ============================================================================
# TRITON KERNELS (Lightning Attention 2)
# ============================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        Out,
        S,
        b: tl.constexpr,
        h: tl.constexpr,
        n: tl.constexpr,
        d: tl.constexpr,
        e: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BLOCK: tl.constexpr,
        BLOCK_MODEL: tl.constexpr,
    ):
        off_bh = tl.program_id(0)
        off_h = off_bh % h
        off_e = tl.program_id(1)
        qk_offset = off_bh * n * d
        v_offset = off_bh * n * e
        o_offset = off_bh * n * e
        e_offset = off_e * BLOCK_MODEL

        Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
        K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
        V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
        O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
        S_block_ptr = S + off_h

        s = tl.load(S_block_ptr)
        off_block = tl.arange(0, BLOCK)
        q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])
        k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))
        block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

        index = off_block[:, None] - off_block[None, :]
        s_index = s * index
        s_index = tl.where(index >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)
        kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

        for i in range(NUM_BLOCK):
            q = tl.load(
                Q_block_ptr + off_block[:, None] * d,
                mask=off_block[:, None] < n,
                other=0.0,
            ).to(tl.float32)
            k_trans = tl.load(
                K_trans_block_ptr + off_block[None, :] * d,
                mask=off_block[None, :] < n,
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V_block_ptr + off_block[:, None] * e,
                mask=off_block[:, None] < n,
                other=0.0,
            ).to(tl.float32)

            qk = tl.dot(q, k_trans) * diag_decay
            o_intra = tl.dot(qk, v)
            o_inter = tl.dot(q, kv) * q_decay
            o = o_intra + o_inter

            tl.store(
                O_block_ptr + off_block[:, None] * e,
                o.to(O_block_ptr.dtype.element_ty),
                mask=off_block[:, None] < n,
            )
            kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
            off_block += BLOCK

    @triton.jit
    def _bwd_intra_kernel(
        Q,
        K,
        V,
        S,
        DO,
        DQ,
        DK,
        DV,
        b: tl.constexpr,
        h: tl.constexpr,
        n: tl.constexpr,
        d: tl.constexpr,
        e: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BLOCK: tl.constexpr,
        CBLOCK: tl.constexpr,
        NUM_CBLOCK: tl.constexpr,
    ):
        off_bh = tl.program_id(0)
        off_block = tl.program_id(1)
        off_h = off_bh % h
        qk_offset = off_bh * n * d
        v_offset = off_bh * n * e
        o_offset = off_bh * n * e
        block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

        Q_trans_block_ptr = (
            Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
        )
        K_block_ptr = (
            K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
        )
        V_trans_block_ptr = (
            V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
        )

        DQ_block_ptr = (
            DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
        )
        DK_trans_block_ptr = (
            DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
        )
        DV_block_ptr = (
            DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
        )
        DO_block_ptr = (
            DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
        )

        S_block_ptr = S + off_h
        s = tl.load(S_block_ptr)
        array = tl.arange(0, BLOCK).to(tl.float32)
        index = array[:, None] - array[None, :]
        s_index = s * index
        s_index = tl.where(index >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)
        diag_decay_trans = tl.trans(diag_decay)

        k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(
            tl.float32
        )
        v_trans = tl.load(
            V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0
        ).to(tl.float32)
        do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(
            tl.float32
        )
        q_trans = tl.load(
            Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0
        ).to(tl.float32)

        dqk = tl.dot(do, v_trans) * diag_decay
        dq = tl.dot(dqk, k)
        dk_trans = tl.dot(q_trans, dqk)
        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv = tl.dot(qk_trans, do)

        tl.store(
            DQ_block_ptr,
            dq.to(DQ_block_ptr.dtype.element_ty),
            mask=block_offset[:, None] < n,
        )
        tl.store(
            DK_trans_block_ptr,
            dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
            mask=block_offset[None, :] < n,
        )
        tl.store(
            DV_block_ptr,
            dv.to(DV_block_ptr.dtype.element_ty),
            mask=block_offset[:, None] < n,
        )

    @triton.jit
    def _bwd_inter_kernel(
        Q,
        K,
        V,
        S,
        DO,
        DQ,
        DK,
        DV,
        b: tl.constexpr,
        h: tl.constexpr,
        n: tl.constexpr,
        d: tl.constexpr,
        e: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BLOCK: tl.constexpr,
        CBLOCK: tl.constexpr,
        NUM_CBLOCK: tl.constexpr,
    ):
        off_bh = tl.program_id(0)
        off_h = off_bh % h
        qk_offset = off_bh * n * d
        v_offset = off_bh * n * e
        o_offset = off_bh * n * e
        S_block_ptr = S + off_h

        DQ_block_ptr = (
            DQ
            + qk_offset
            + tl.arange(0, CBLOCK)[:, None] * d
            + tl.arange(0, d)[None, :]
        )
        K_block_ptr = (
            K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
        )
        V_trans_block_ptr = (
            V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
        )
        DO_block_ptr = (
            DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
        )
        off_block1, off_block2 = tl.arange(0, CBLOCK), tl.arange(0, CBLOCK)
        c_array = tl.arange(0, CBLOCK)

        s = tl.load(S_block_ptr)
        block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
        kv_trans = tl.zeros([e, d], dtype=tl.float32)

        for i in range(NUM_BLOCK):
            for j in range(NUM_CBLOCK):
                if i > 0:
                    q_decay = tl.exp(
                        -s.to(tl.float32) * (j * CBLOCK + c_array[:, None])
                    )
                    do = tl.load(
                        DO_block_ptr, mask=off_block1[:, None] < n, other=0.0
                    ).to(tl.float32)
                    dq = tl.dot(do, kv_trans) * q_decay + tl.load(
                        DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                    )
                    tl.store(
                        DQ_block_ptr,
                        dq.to(DQ_block_ptr.dtype.element_ty),
                        mask=off_block1[:, None] < n,
                    )
                DQ_block_ptr += CBLOCK * d
                DO_block_ptr += CBLOCK * e
                off_block1 += CBLOCK

            kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
            for j in range(NUM_CBLOCK):
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
                ).to(tl.float32)
                k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                    tl.float32
                )
                k_decay = tl.exp(
                    -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
                )
                kv_trans_current += tl.dot(v_trans, k * k_decay)
                K_block_ptr += CBLOCK * d
                V_trans_block_ptr += CBLOCK * e
                off_block2 += CBLOCK
            kv_trans = block_decay * kv_trans + kv_trans_current

        m = NUM_BLOCK * BLOCK
        off_block1, off_block2 = m + tl.arange(0, CBLOCK), m + tl.arange(0, CBLOCK)
        Q_trans_block_ptr = (
            Q
            + qk_offset
            + m * d
            + tl.arange(0, CBLOCK)[None, :] * d
            + tl.arange(0, d)[:, None]
        )
        K_block_ptr = (
            K
            + qk_offset
            + m * d
            + tl.arange(0, CBLOCK)[:, None] * d
            + tl.arange(0, d)[None, :]
        )
        V_trans_block_ptr = (
            V
            + v_offset
            + m * e
            + tl.arange(0, CBLOCK)[None, :] * e
            + tl.arange(0, e)[:, None]
        )
        DK_trans_block_ptr = (
            DK
            + qk_offset
            + m * d
            + tl.arange(0, CBLOCK)[None, :] * d
            + tl.arange(0, d)[:, None]
        )
        DV_block_ptr = (
            DV
            + v_offset
            + m * e
            + tl.arange(0, CBLOCK)[:, None] * e
            + tl.arange(0, e)[None, :]
        )
        DO_block_ptr = (
            DO
            + o_offset
            + m * e
            + tl.arange(0, CBLOCK)[:, None] * e
            + tl.arange(0, e)[None, :]
        )

        dkv = tl.zeros([d, e], dtype=tl.float32)
        for i in range(NUM_BLOCK - 1, -1, -1):
            for j in range(NUM_CBLOCK - 1, -1, -1):
                K_block_ptr -= CBLOCK * d
                V_trans_block_ptr -= CBLOCK * e
                DK_trans_block_ptr -= CBLOCK * d
                DV_block_ptr -= CBLOCK * e
                off_block1 -= CBLOCK
                if i < NUM_BLOCK - 1:
                    k = tl.load(
                        K_block_ptr, mask=off_block1[:, None] < n, other=0.0
                    ).to(tl.float32)
                    v_trans = tl.load(
                        V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                    ).to(tl.float32)
                    k_decay_trans = tl.exp(
                        -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :]))
                    )
                    k_decay = tl.exp(
                        -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
                    )
                    dk_trans = tl.dot(dkv, v_trans) * k_decay_trans + tl.load(
                        DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                    )
                    dv = tl.dot(k, dkv) * k_decay + tl.load(
                        DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                    )
                    tl.store(
                        DK_trans_block_ptr,
                        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                        mask=off_block1[None, :] < n,
                    )
                    tl.store(
                        DV_block_ptr,
                        dv.to(DV_block_ptr.dtype.element_ty),
                        mask=off_block1[:, None] < n,
                    )

            dkv_current = tl.zeros([d, e], dtype=tl.float32)
            for j in range(NUM_CBLOCK - 1, -1, -1):
                DO_block_ptr -= CBLOCK * e
                Q_trans_block_ptr -= CBLOCK * d
                off_block2 -= CBLOCK
                do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                    tl.float32
                )
                q_trans = tl.load(
                    Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
                ).to(tl.float32)
                q_decay_trans = tl.exp(
                    -s.to(tl.float32) * (j * CBLOCK + c_array[None, :])
                )
                dkv_current += tl.dot(q_trans * q_decay_trans, do)
            dkv = block_decay * dkv + dkv_current

    class LightningAttention2Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, s):
            q, k, v, s = q.contiguous(), k.contiguous(), v.contiguous(), s.contiguous()
            b, h, n, d = q.shape
            e = v.shape[-1]
            o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
            BLOCK = 64
            NUM_BLOCK = triton.cdiv(n, BLOCK)
            BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
            grid = (b * h, triton.cdiv(e, BLOCK_MODEL))
            _fwd_kernel[grid](
                q,
                k,
                v,
                o,
                s,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                BLOCK_MODEL=BLOCK_MODEL,
            )
            ctx.save_for_backward(q, k, v, s)
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, s = ctx.saved_tensors
            q, k, v, s, do = (
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                s.contiguous(),
                do.contiguous(),
            )
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            b, h, n, d = q.shape
            e = v.shape[-1]
            BLOCK, CBLOCK = 64, 32
            NUM_BLOCK, NUM_CBLOCK = triton.cdiv(n, BLOCK), BLOCK // CBLOCK
            grid_intra = (b * h, NUM_BLOCK)
            _bwd_intra_kernel[grid_intra](
                q,
                k,
                v,
                s,
                do,
                dq,
                dk,
                dv,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
            grid_inter = (b * h,)
            _bwd_inter_kernel[grid_inter](
                q,
                k,
                v,
                s,
                do,
                dq,
                dk,
                dv,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
            return dq, dk, dv, None

    lightning_attn2_fn = LightningAttention2Function.apply
else:
    lightning_attn2_fn = None


# ============================================================================
# MODERN TRANSFORMER COMPONENTS
# ============================================================================


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    B, N, H, D = x.shape
    cos_slice = cos[:N].view(1, N, 1, D)
    sin_slice = sin[:N].view(1, N, 1, D)
    x_half = x.reshape(B, N, H, D // 2, 2)
    x_rotated = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).reshape(
        B, N, H, D
    )
    return x * cos_slice + x_rotated * sin_slice


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return w + (((w * scale).round().clamp(-1, 1) / scale) - w).detach()


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.RMSNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(self.norm(x), weight_quant(self.weight), self.bias)


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = BitLinear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        ) * self.weight


# ============================================================================
# DEEPSEEK V4: REAL ENGRAMS
# ============================================================================


class EngramLayer(nn.Module):
    def __init__(self, dim, vocab_size=370000, num_heads=4):
        super().__init__()
        self.memory_table = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.memory_table.weight, std=0.02)
        self.W_K = QuantizedLinear(dim, dim)
        self.W_V = QuantizedLinear(dim, dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, h_t, hash_ngrams):
        B, L, D = h_t.shape
        e_t = self.memory_table(hash_ngrams).sum(dim=2)
        e_conv = self.conv(e_t.transpose(1, 2)).transpose(1, 2)[:, :L]
        q_norm = F.rms_norm(h_t, (D,))
        k_norm = F.rms_norm(self.W_K(e_conv), (D,))
        alpha = torch.sigmoid((q_norm * k_norm).sum(dim=-1, keepdim=True))
        return h_t + alpha * self.W_V(e_conv)


# ============================================================================
# DEEPSEEK V3: mHC
# ============================================================================


class ManifoldHyperConnection(nn.Module):
    def __init__(self, num_streams, num_iters=3):
        super().__init__()
        self.n = num_streams
        self.num_iters = num_iters
        self.raw_weight = nn.Parameter(torch.randn(num_streams, num_streams) * 0.02)

    def sinkhorn_knopp(self, w):
        M = torch.exp(w)
        for _ in range(self.num_iters):
            M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
            M = M / (M.sum(dim=0, keepdim=True) + 1e-6)
        return M

    def precompute(self):
        self.register_buffer("cached_H", self.sinkhorn_knopp(self.raw_weight))

    def forward(self, x):
        B, L, D = x.shape
        stream_dim = D // self.n
        x_streams = x.view(B, L, self.n, stream_dim)
        H_res = getattr(self, "cached_H", self.sinkhorn_knopp(self.raw_weight))
        return torch.einsum("blnd,nm->blmd", x_streams, H_res).reshape(B, L, D)


# ============================================================================
# TRANSFORMER: LIGHTNING ATTENTION 2
# ============================================================================


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * dim / 3)
        self.w1, self.w2, self.w3 = (
            QuantizedLinear(dim, hidden_dim),
            QuantizedLinear(hidden_dim, dim),
            QuantizedLinear(dim, hidden_dim),
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LightningAttention2(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = QuantizedLinear(dim, dim * 3, bias=False)
        self.proj = QuantizedLinear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.s = nn.Parameter(torch.full((num_heads,), 0.02))

    def forward(self, x, cos=None, sin=None, mask=None, kv_cache=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        if cos is not None and sin is not None:
            offset = kv_cache[1] if kv_cache is not None else 0
            q = apply_rope(q, cos[offset : offset + N], sin[offset : offset + N])
            k = apply_rope(k, cos[offset : offset + N], sin[offset : offset + N])

        if kv_cache is not None:
            prev_s, offset = kv_cache
        else:
            prev_s = torch.zeros(
                B,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )
            offset = 0

        if TRITON_AVAILABLE and N > 1 and not torch.jit.is_scripting():
            out = lightning_attn2_fn(q, k, v, self.s)
            # Recursive update for state (vectorized loop)
            decay = (-self.s).exp().view(1, self.num_heads, 1, 1)
            curr_s = prev_s
            for i in range(N):
                curr_s = decay * (
                    curr_s + torch.einsum("bhd,bhe->bhde", k[:, i], v[:, i])
                )
        else:
            decay = (-self.s).exp().view(1, self.num_heads, 1, 1)
            out_list = []
            curr_s = prev_s
            for i in range(N):
                ki, vi = k[:, i], v[:, i]
                # Formula matching Triton for N=1: o = q @ (kv + k^T v); new_kv = exp(-s) * (kv + k^T v)
                state_with_current = curr_s + torch.einsum("bhd,bhe->bhde", ki, vi)
                oi = torch.einsum(
                    "bhd,bhde->bhe", q[:, i] * self.scale, state_with_current
                )
                out_list.append(oi)
                curr_s = decay * state_with_current
            out = torch.stack(out_list, dim=1)

        return self.proj(self.norm(out)), (curr_s, offset + N)


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1, self.attn, self.mhc_attn = (
            RMSNorm(dim),
            LightningAttention2(dim, num_heads),
            ManifoldHyperConnection(16),
        )
        self.norm2, self.mlp, self.mhc_mlp = (
            RMSNorm(dim),
            SwiGLUMLP(dim),
            ManifoldHyperConnection(16),
        )

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        # Attention block with mHC mixing
        x_mixed = self.mhc_attn(x)
        attn_out, next_kv = self.attn(self.norm1(x), cos, sin, mask, kv_cache)
        x = x_mixed + attn_out

        # MLP block with mHC mixing
        x = self.mhc_mlp(x) + self.mlp(self.norm2(x))
        return x, next_kv


# ============================================================================
# MAIN MODEL: ULTRA-EFFICIENT TEXT JEPA
# ============================================================================


class UltraEfficientTextJEPA(nn.Module):
    def __init__(
        self,
        vocab_size=100277,
        source_dim=512,
        source_depth=24,
        engram_vocab_size=370000,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, source_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        self.engram = EngramLayer(source_dim, vocab_size=engram_vocab_size)
        self.blocks = nn.ModuleList(
            [Transformer(source_dim, num_heads=16) for _ in range(source_depth)]
        )
        self.norm = RMSNorm(source_dim)
        self.lm_head = QuantizedLinear(source_dim, vocab_size)
        self.lm_head.linear.weight = self.token_embed.weight
        cos, sin = precompute_rope_freqs(source_dim // 16, 4096)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def precompute_mhc(self):
        for b in self.blocks:
            b.mhc_attn.precompute()
            b.mhc_mlp.precompute()

    def forward(
        self,
        source_tokens,
        target_tokens=None,
        hash_ngrams=None,
        return_embeddings=False,
        past_key_values=None,
        **kwargs,
    ):
        x = self.token_embed(source_tokens)
        if hash_ngrams is not None:
            x = self.engram(x, hash_ngrams)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, layer_kv = block(
                x,
                self.cos,
                self.sin,
                kv_cache=(past_key_values[i] if past_key_values else None),
            )
            new_kvs.append(layer_kv)
        logits = self.lm_head(self.norm(x))
        if target_tokens is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                target_tokens[:, 1:].reshape(-1),
            )
            return loss, logits, new_kvs
        return (x if return_embeddings else logits), new_kvs
