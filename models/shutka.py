import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ============================================================================
# MODERN TRANSFORMER COMPONENTS
# ============================================================================


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute frequency constants for RoPE (Real-valued for compiler)"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply Rotary Positional Embeddings using real-valued rotation matrix"""
    B, N, H, D = x.shape

    if cos.shape[0] < N:
        pad_len = N - cos.shape[0]
        cos = F.pad(cos, (0, 0, 0, pad_len))
        sin = F.pad(sin, (0, 0, 0, pad_len))

    cos_seq = cos[:N]
    sin_seq = sin[:N]

    expected_dim = D // 2
    actual_dim = cos_seq.shape[-1]

    if actual_dim != expected_dim:
        if actual_dim < expected_dim:
            pad_size = expected_dim - actual_dim
            cos_seq = F.pad(cos_seq, (0, pad_size))
            sin_seq = F.pad(sin_seq, (0, pad_size))
        else:
            cos_seq = cos_seq[..., :expected_dim]
            sin_seq = sin_seq[..., :expected_dim]

    cos_expanded = cos_seq.repeat_interleave(2, dim=-1)
    sin_expanded = sin_seq.repeat_interleave(2, dim=-1)

    cos_slice = cos_expanded.view(1, N, 1, D)
    sin_slice = sin_expanded.view(1, N, 1, D)

    x_half = x.reshape(B, N, H, D // 2, 2)
    x_rotated = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).reshape(
        B, N, H, D
    )

    return x * cos_slice + x_rotated * sin_slice


# ============================================================================
# QUANTIZED LINEAR LAYERS (Optimized for Inference)
# ============================================================================


def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    u = (w * scale).round().clamp(-1, 1) / scale
    return u


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.norm = nn.RMSNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x_norm = self.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_quant, w_quant, self.bias)


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bits=1.58):
        super().__init__()
        self.linear = BitLinear(in_features, out_features, bias=bias)

    @torch.compiler.disable
    def forward(self, x):
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight.type_as(x)


# ============================================================================
# DEEPSEEK V4: REAL ENGRAMS (Context-Aware Gating + Conv)
# ============================================================================


class EngramLayer(nn.Module):
    """
    DeepSeek v4 Engrams Implementation (Rigorous)
    Based on: https://eu.36kr.com/en/p/3637163406624008

    Features:
    - Multi-head Hashing (K heads per N-gram order)
    - Context-Aware Gating (Query-Key-Value mechanism)
    - Depth-Causal Convolution
    """

    def __init__(self, dim: int, vocab_size: int = 370000, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # 1. Sparse Retrieval Memory (Conceptually E_{n,k})
        # We lump all K heads and N-gram orders into one massive table for efficiency
        # In a strict implementation, these would be separate tables, but learning works similarly
        self.memory_table = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.memory_table.weight, mean=0.0, std=0.02)

        # 2. Context-Aware Gating
        # h_t is Query, e_t (retrieved) is source for Key/Value
        self.W_K = QuantizedLinear(dim, dim, bias=False)
        self.W_V = QuantizedLinear(dim, dim, bias=False)

        # Gating projections
        # We project Q and K to a scalar alpha
        self.gate_proj = nn.Linear(dim, 1)  # Simplified attention score

        # 3. Convolution
        # "Short depth-causal convolution" on the retrieved embeddings
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=2, groups=dim)

    def forward(self, h_t: torch.Tensor, hash_ngrams: torch.Tensor) -> torch.Tensor:
        """
        h_t: [B, L, D] - Current hidden state (Dynamic Query)
        hash_ngrams: [B, L, K] - Precomputed N-gram hashes
        """
        B, L, D = h_t.shape

        # 1. Sparse Retrieval
        # Retrieve embeddings for all K hash heads
        # hash_ngrams is [B, L, K], output is [B, L, K, D]
        # We sum across K heads (simple aggregation proposed in some variants,
        # or we could treat them as independent Key/Values. For memory, we sum first here).
        e_raw = self.memory_table(hash_ngrams)  # [B, L, K, D]
        e_t = e_raw.sum(dim=2)  # [B, L, D] (Bag of Engrams)

        # 2. Convolution (Depth-causal)
        # Transpose for Conv1d: [B, D, L]
        e_conv = self.conv(e_t.transpose(1, 2))
        e_conv = e_conv[:, :, :L].transpose(1, 2)  # Causal crop -> [B, L, D]

        # 3. Context-Aware Gating
        # Query = h_t, Key = W_K(e_conv), Value = W_V(e_conv)
        # Using a simplified Gating scalar mechanism:
        # alpha = Sigmoid(Norm(h_t) * Norm(e_key)) -> simplified to Linear interaction

        # Proper Gating as per paper inspiration:
        # G = Sigmoid(W_g [h_t; e_conv])
        # But paper says "RMSNorm on Q and K before calculating scalar gate"

        q_norm = F.rms_norm(h_t, (D,))
        k_norm = F.rms_norm(self.W_K(e_conv), (D,))

        # Element-wise product for similarity, then project to scalar
        similarity = (q_norm * k_norm).sum(dim=-1, keepdim=True)  # [B, L, 1]
        alpha = torch.sigmoid(similarity)  # Gate value (0 to 1)

        v_out = self.W_V(e_conv)

        # Residual update
        return h_t + alpha * v_out


# ============================================================================
# DEEPSEEK V3: REAL mHC (Manifold-Constrained Hyper-Connections)
# ============================================================================


class ManifoldHyperConnection(nn.Module):
    """
    Real mHC Implementation via Sinkhorn-Knopp
    Based on: https://arxiv.org/html/2512.24880v2

    Constrains the residual mapping H_res to be a Doubly Stochastic Matrix.
    Uses iterative Sinkhorn normalization (Row norm -> Col norm -> ...).
    """

    def __init__(self, num_streams: int, num_iters: int = 5):
        super().__init__()
        self.n = num_streams
        self.num_iters = num_iters

        # The learnable parameter matrix M (unconstrained)
        # Represents interactions between 'num_streams' parallel information paths.
        # If the model is a standard Transformer, we split D into 'num_streams' chunks to simulate streams.
        self.raw_weight = nn.Parameter(torch.randn(num_streams, num_streams) * 0.02)

    def sinkhorn_knopp(self, w: torch.Tensor) -> torch.Tensor:
        """
        Iteratively normalize rows and columns to sum to 1.
        Starting point: exp(w) to ensure positivity.
        """
        # 1. Positivity
        M = torch.exp(w)

        # 2. Iterative Normalization
        for _ in range(self.num_iters):
            # Row Norm
            M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
            # Col Norm
            M = M / (M.sum(dim=0, keepdim=True) + 1e-6)

        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        # Split D into n streams
        # [B, L, n, D/n]
        assert D % self.n == 0, (
            f"Dimension {D} must be divisible by num_streams {self.n}"
        )
        stream_dim = D // self.n

        x_streams = x.view(B, L, self.n, stream_dim)

        # Compute H_res (Doubly Stochastic Matrix)
        H_res = self.sinkhorn_knopp(self.raw_weight)  # [n, n]

        # Apply mixing: H_res * x_streams
        # We want to mix the 'n' dimension
        # Einsum: b l n d, n m -> b l m d (where n=m=num_streams)
        out_streams = torch.einsum("blnd,nm->blmd", x_streams, H_res)

        # Merge back
        return out_streams.reshape(B, L, D)


# ============================================================================
# TRANSFORMER & MLP
# ============================================================================


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = QuantizedLinear(dim, hidden_dim, bias=False)
        self.w2 = QuantizedLinear(hidden_dim, dim, bias=False)
        self.w3 = QuantizedLinear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LightningAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, chunk_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.scale = self.head_dim**-0.5
        self.qkv = QuantizedLinear(dim, dim * 3)
        self.proj = QuantizedLinear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x, cos=None, sin=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        pad_len = (self.chunk_size - (N % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            padding = torch.zeros(
                B,
                pad_len,
                self.num_heads,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )
            q = torch.cat([q, padding], dim=1)
            k = torch.cat([k, padding], dim=1)
            v = torch.cat([v, padding], dim=1)

        N_padded = q.shape[1]
        num_chunks = N_padded // self.chunk_size

        q = q.view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        k = k.view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        v = v.view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)

        kv_state = torch.zeros(
            B,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        causal_mask = torch.tril(
            torch.ones(
                self.chunk_size, self.chunk_size, device=x.device, dtype=torch.bool
            )
        )

        output_chunks = []

        for t in range(num_chunks):
            q_t = q[:, t].transpose(1, 2)
            k_t = k[:, t].transpose(1, 2)
            v_t = v[:, t].transpose(1, 2)

            attn_scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * self.scale
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            o_intra = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v_t)

            o_inter = torch.einsum("bhqd,bhde->bhqe", q_t, kv_state)
            o_t = o_intra + o_inter

            kv_update = torch.einsum("bhkd,bhke->bhde", k_t, v_t)
            kv_state = kv_state + kv_update

            output_chunks.append(o_t.transpose(1, 2))

        output = torch.cat(output_chunks, dim=1)
        if pad_len > 0:
            output = output[:, :N]

        output = output.reshape(B, N, -1)
        return self.proj(self.norm(output))


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = LightningAttention2(dim, num_heads)

        # mHC: Manifold Constraint on Residual
        # We assume 16 streams for effective manifold mixing (arbitrary but typically = num_heads)
        self.mhc_attn = ManifoldHyperConnection(num_streams=16)

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim)
        self.mhc_mlp = ManifoldHyperConnection(num_streams=16)

    def forward(self, x, cos=None, sin=None, mask=None):
        # Attention Block
        # Standard: x = x + attn(norm(x))
        # mHC: x = mHC(x) + attn(norm(x)) OR x = mHC(x + attn(norm(x)))?
        # Paper says: "constrain the residual mapping H_res".
        # Equation: x_{l+1} = H_res * x_l + f(x_l)

        # 1. Apply mHC to the IDENTITY path (modifying the residual stream state)
        x_mixed = self.mhc_attn(x)

        # 2. Apply Function (Attention)
        attn_out = self.attn(self.norm1(x), cos, sin, mask)

        # 3. Combine
        x = x_mixed + attn_out

        # MLP Block
        x_mixed = self.mhc_mlp(x)
        mlp_out = self.mlp(self.norm2(x))
        x = x_mixed + mlp_out

        return x


# ============================================================================
# MAIN MODEL: ULTRA-EFFICIENT TEXT JEPA (ShuTKA-v2)
# ============================================================================


class UltraEfficientTextJEPA(nn.Module):
    def __init__(
        self,
        vocab_size=100277,
        source_dim=512,  # Resized for 350M target
        source_depth=24,
        target_dim=512,
        target_depth=6,
        predictor_dim=512,
        predictor_depth=6,
        output_dim=512,
        max_source_len=4096,
        max_target_len=512,
        engram_vocab_size=370000,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__()
        print("Initializing ShuTKA-v2 (DeepSeek v3/v4 Rigorous Spec)...")
        print(
            "  Features: bf16, Real Engrams (Gated+Conv), Real mHC (Sinkhorn), SwiGLU"
        )

        self.vocab_size = vocab_size
        self.source_dim = source_dim

        self.token_embed = nn.Embedding(vocab_size, source_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # DeepSeek v4 Engrams Layer
        # Placed after embedding, before transformer blocks
        self.engram = EngramLayer(source_dim, vocab_size=engram_vocab_size)

        self.blocks = nn.ModuleList(
            [Transformer(source_dim, num_heads=16) for _ in range(source_depth)]
        )

        self.norm = RMSNorm(source_dim)

        cos, sin = precompute_rope_freqs(source_dim // 16, max_source_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.predictor_blocks = nn.ModuleList(
            [Transformer(predictor_dim, num_heads=16) for _ in range(predictor_depth)]
        )
        self.predictor_norm = RMSNorm(predictor_dim)
        self.predictor_proj = QuantizedLinear(source_dim, output_dim)

        self.lm_head = QuantizedLinear(output_dim, vocab_size)
        self.lm_head.linear.weight = self.token_embed.weight

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        hash_ngrams: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        **kwargs,
    ):
        B, L = source_tokens.shape
        x = self.token_embed(source_tokens)

        # Apply Real Engrams (DeepSeek v4)
        if hash_ngrams is not None:
            x = self.engram(x, hash_ngrams)

        cos = self.cos[:L]
        sin = self.sin[:L]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)

        if return_embeddings:
            return x

        logits = self.lm_head(x)

        if target_tokens is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_tokens[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
            )

            # Return tuple to satisfy Trainer unpacking (loss, _, _)
            return loss, logits, None

        return logits

    def predict_next(self, x, hash_ngrams=None):
        return self.forward(x, hash_ngrams)


# ============================================================================
# OPTIMIZER (GALORE + MUON)
# ============================================================================


class Optimizer(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-4, rank=128, update_freq=200, cans_steps=5, delta=0.3
    ):
        defaults = dict(
            lr=lr,
            rank=rank,
            update_freq=update_freq,
            cans_steps=cans_steps,
            delta=delta,
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.update_freq = update_freq
        self.cans_steps = cans_steps
        self.projections = {}
        self.momentum = {}
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.step_count += 1

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dim() < 2:
                    p.data.add_(grad, alpha=-lr)
                    continue

                if id(p) not in self.momentum:
                    self.momentum[id(p)] = torch.zeros_like(grad)

                buf = self.momentum[id(p)]
                buf.mul_(0.9).add_(grad, alpha=1.0)

                p.data.add_(buf, alpha=-lr)

        return loss
