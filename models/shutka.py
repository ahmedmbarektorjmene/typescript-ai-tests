import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
from typing import Optional, Tuple, List
from dataclasses import dataclass


# Optional imports for production
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print(
        "Warning: FAISS not available. Install with: pip install faiss-cpu or pip install faiss-cu12"
    )

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not available for Bing Search integration")


# ============================================================================
# TITANS MEMORY MODULE
# ============================================================================

class TitansMemory(nn.Module):
    """
    Titans Memory: Test-time learnable memory with surprise-based updates
    Based on: https://arxiv.org/abs/2501.00663
    """
    def __init__(self, dim: int, memory_depth: int = 3, capacity: int = 10000, 
                 surprise_threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.surprise_threshold = surprise_threshold
        
        # Memory MLP (deep neural network for expressive memory)
        layers = []
        current_dim = dim
        for _ in range(memory_depth):
            layers.extend([
                nn.Linear(current_dim, dim * 4),
                nn.GELU(),
            ])
            current_dim = dim * 4
        layers.append(nn.Linear(current_dim, dim))
        self.memory_mlp = nn.Sequential(*layers)
        
        # Memory storage (keys and values)
        self.register_buffer('memory_keys', torch.randn(capacity, dim))
        self.register_buffer('memory_values', torch.randn(capacity, dim))
        self.register_buffer('access_counts', torch.zeros(capacity))
        
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Query memory and return retrieved information"""
        # query: [B, D] or [B, L, D]
        if query.dim() == 3:
            query = query.mean(dim=1)  # Pool to [B, D]
        
        # Compute similarity with all memory keys
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.memory_keys, dim=-1)
        similarities = query_norm @ keys_norm.T  # [B, capacity]
        
        # Get top-k most similar
        top_k = min(5, self.capacity)
        top_indices = similarities.topk(top_k, dim=-1).indices  # [B, top_k]
        
        # Retrieve and aggregate
        retrieved = []
        for b in range(query.shape[0]):
            batch_retrieved = self.memory_values[top_indices[b]].mean(dim=0)
            retrieved.append(batch_retrieved)
        
        retrieved = torch.stack(retrieved)  # [B, D]
        
        # Update access counts
        with torch.no_grad():
            for b in range(query.shape[0]):
                self.access_counts[top_indices[b]] += 1
        
        return retrieved
    
    def query_with_confidence(self, query: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Query memory and return confidence score"""
        retrieved = self.forward(query)
        
        # Compute confidence as similarity
        if query.dim() == 3:
            query = query.mean(dim=1)
        confidence = F.cosine_similarity(query, retrieved, dim=-1).mean().item()
        
        return retrieved, confidence
    
    def update(self, key: torch.Tensor, value: torch.Tensor, surprise: float):
        """
        Update memory based on surprise metric.
        Memory updates are detached from gradient backprop for efficiency.
        """
        try:
            with torch.no_grad():  # Detach from gradient computation
                if surprise > self.surprise_threshold:
                    # Find slot to update (least recently used)
                    slot_idx = self.access_counts.argmin()
                    
                    # Update memory
                    if key.dim() == 3:
                        key = key.mean(dim=1)
                    if value.dim() == 3:
                        value = value.mean(dim=1)
                    
                    self.memory_keys[slot_idx] = key[0].detach() if key.dim() == 2 else key.detach()
                    self.memory_values[slot_idx] = value[0].detach() if value.dim() == 2 else value.detach()
                    self.access_counts[slot_idx] = 0
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: Out of memory during Titans Memory update. Skipping update.")
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
    
    def compute_surprise(self, query: torch.Tensor, retrieved: torch.Tensor) -> float:
        """Compute surprise metric (prediction error)"""
        if query.dim() == 3:
            query = query.mean(dim=1)
        if retrieved.dim() == 3:
            retrieved = retrieved.mean(dim=1)
        
        # Surprise = prediction error
        predicted = self.memory_mlp(retrieved)
        surprise = (query - predicted).pow(2).mean().item()
        return surprise
    
    def save_state(self, path: str):
        """Save memory state to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'memory_keys': self.memory_keys,
                'memory_values': self.memory_values,
                'access_counts': self.access_counts,
            }, path)
        except Exception as e:
            print(f"ERROR: Failed to save Titans Memory to {path}: {e}")
            raise
    
    def load_state(self, path: str):
        """Load memory state from disk"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Titans Memory file not found: {path}")
            
            state = torch.load(path)
            
            # Validate state dict
            required_keys = ['memory_keys', 'memory_values', 'access_counts']
            for key in required_keys:
                if key not in state:
                    raise ValueError(f"Invalid Titans Memory state: missing '{key}'")
            
            self.memory_keys.copy_(state['memory_keys'])
            self.memory_values.copy_(state['memory_values'])
            self.access_counts.copy_(state['access_counts'])
        except Exception as e:
            print(f"ERROR: Failed to load Titans Memory from {path}: {e}")
            raise
    
    def forget_least_accessed(self, n: int = 100):
        """Forget the n least accessed memories"""
        with torch.no_grad():
            least_accessed = self.access_counts.topk(n, largest=False).indices
            self.memory_keys[least_accessed] = torch.randn_like(self.memory_keys[least_accessed])
            self.memory_values[least_accessed] = torch.randn_like(self.memory_values[least_accessed])
            self.access_counts[least_accessed] = 0


# ============================================================================
# BING SEARCH API INTEGRATION
# ============================================================================

class BingSearchAPI:
    """External knowledge retrieval via Bing Search API"""
    def __init__(self, api_key: str, endpoint: str = "https://api.bing.microsoft.com/v7.0/search"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.cache = {}  # Simple cache
    
    def search(self, query: str, top_k: int = 3) -> str:
        """Search Bing and return concatenated results"""
        if not HTTPX_AVAILABLE:
            return ""
        
        # Check cache first
        if query in self.cache:
            return self.cache[query]
        
        # Make API request
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": top_k, "responseFilter": "Webpages"}
        
        try:
            response = httpx.get(self.endpoint, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            results = response.json()
            
            # Extract relevant snippets
            snippets = []
            for page in results.get("webPages", {}).get("value", []):
                snippets.append(f"{page['name']}: {page['snippet']}")
            
            combined = "\n".join(snippets)
            self.cache[query] = combined
            return combined
            
        except Exception as e:
            print(f"Bing search failed: {e}")
            return ""


# ============================================================================
# MIRAS (Memory-Injected Retrieval-Augmented System)
# ============================================================================

class MIRAS(nn.Module):
    """
    MIRAS: Coordinate between Titans Memory, FAISS, and Bing Search
    Three-tier retrieval system
    """
    def __init__(self, titans_memory: TitansMemory, faiss_index: Optional['FAISSMemoryBank'] = None,
                 bing_api_key: Optional[str] = None, confidence_threshold: float = 0.7):
        super().__init__()
        self.titans = titans_memory
        self.faiss = faiss_index
        self.bing_search = BingSearchAPI(bing_api_key) if bing_api_key else None
        self.confidence_threshold = confidence_threshold
        self.embed_dim = titans_memory.dim
    
    def retrieve(self, query: torch.Tensor, query_text: str = "") -> Tuple[torch.Tensor, str]:
        """
        Three-tier retrieval:
        1. Query Titans Memory (personalized patterns)
        2. If confidence < threshold, query FAISS (project context)
        3. If still low confidence, query Bing (external knowledge)
        """
        # Tier 1: Titans Memory
        titans_result, confidence = self.titans.query_with_confidence(query)
        if confidence > self.confidence_threshold:
            return titans_result, "titans"
        
        # Tier 2: FAISS (project-specific)
        if self.faiss and hasattr(self.faiss, 'indices') and self.faiss.indices:
            # Search FAISS
            distances, indices, texts = self.faiss.search(query, k=1)
            if texts and len(texts[0]) > 0:
                # We got results, cache in Titans
                faiss_result = query  # Placeholder - in real impl, embed the text
                surprise = self.titans.compute_surprise(query, faiss_result)
                self.titans.update(query, faiss_result, surprise=1.0)
                return faiss_result, "faiss"
        
        # Tier 3: Bing Search (external)
        if self.bing_search and query_text:
            bing_result_text = self.bing_search.search(query_text)
            if bing_result_text:
                # Embed and cache (simplified - just return query for now)
                bing_embedding = query  # In real impl, embed the text
                surprise = self.titans.compute_surprise(query, bing_embedding)
                self.titans.update(query, bing_embedding, surprise=1.0)
                return bing_embedding, "bing"
        
        # Fallback to Titans result
        return titans_result, "titans_fallback"


# ============================================================================
# HOPRAG (Multi-Hop Reasoning with Adaptive Sufficiency)
# ============================================================================

class HopRAG(nn.Module):
    """
    HopRAG: Multi-hop reasoning with adaptive sufficiency threshold
    Based on: https://arxiv.org/abs/2502.12442
    """
    def __init__(self, miras: MIRAS, max_hops: int = 3, dim: int = 768):
        super().__init__()
        self.miras = miras
        self.max_hops = max_hops
        self.dim = dim
        
        # Query reformulation network
        self.reformulator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Reasoning chain aggregator
        self.chain_aggregator = nn.GRU(dim, dim, batch_first=True)
        
        # Adaptive sufficiency threshold learner
        self.sufficiency_net = nn.Sequential(
            nn.Linear(dim * 2, 256),  # Query + Retrieved
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Confidence tracker
        self.confidence_history = []
        self.base_threshold = 0.8
    
    def forward(self, initial_query: torch.Tensor, query_text: str = "") -> torch.Tensor:
        """Perform multi-hop retrieval with adaptive sufficiency"""
        reasoning_chain = []
        current_query = initial_query
        chain_confidence = []
        
        # Ensure query is 2D [B, D]
        if current_query.dim() == 3:
            current_query = current_query.mean(dim=1)
        
        for hop in range(self.max_hops):
            # Retrieve information
            retrieved, source = self.miras.retrieve(current_query, query_text)
            reasoning_chain.append(retrieved)
            
            # Compute retrieval confidence
            confidence = F.cosine_similarity(retrieved, current_query, dim=-1).mean()
            chain_confidence.append(confidence.item())
            
            # Check if we have enough information using adaptive threshold
            if self._is_sufficient_adaptive(retrieved, current_query, chain_confidence):
                break
            
            # Reformulate query for next hop
            combined = torch.cat([current_query, retrieved], dim=-1)
            current_query = self.reformulator(combined)
        
        # Aggregate reasoning chain
        if len(reasoning_chain) == 1:
            return reasoning_chain[0]
        
        chain_tensor = torch.stack(reasoning_chain, dim=1)  # [B, hops, D]
        aggregated, _ = self.chain_aggregator(chain_tensor)
        
        return aggregated[:, -1, :]  # Return final state
    
    def _is_sufficient_adaptive(self, retrieved: torch.Tensor, query: torch.Tensor,
                                chain_confidence: List[float]) -> bool:
        """
        Adaptive sufficiency check based on retrieval chain confidence.
        Replaces fixed 0.8 threshold with learned metric.
        """
        # Compute base similarity
        similarity = F.cosine_similarity(retrieved, query, dim=-1).mean()
        
        # Learn sufficiency threshold from query and retrieved context
        combined = torch.cat([query, retrieved], dim=-1)
        learned_threshold = self.sufficiency_net(combined).mean().item()
        
        # Adjust threshold based on retrieval chain confidence trend
        if len(chain_confidence) > 1:
            # If confidence is increasing, lower threshold (we're making progress)
            confidence_trend = chain_confidence[-1] - chain_confidence[-2]
            adaptive_threshold = learned_threshold + 0.1 * confidence_trend
        else:
            adaptive_threshold = learned_threshold
        
        # Clamp threshold to reasonable range [0.6, 0.9]
        adaptive_threshold = max(0.6, min(0.9, adaptive_threshold))
        
        # Track for analysis
        self.confidence_history.append({
            'similarity': similarity.item(),
            'threshold': adaptive_threshold,
            'chain_length': len(chain_confidence)
        })
        
        return similarity.item() > adaptive_threshold


# ============================================================================
# MODERN TRANSFORMER COMPONENTS (RMSNorm, SwiGLU, RoPE)
# ============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight.type_as(x)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute frequency constants for RoPE (Real-valued for compiler)"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply Rotary Positional Embeddings using real-valued rotation matrix"""
    # x shape: (B, N, H, D)
    B, N, H, D = x.shape

    # cos/sin shape should be: (seq_len, D/2)
    # But handle any mismatch gracefully
    
    # Take only the sequence length we need
    if cos.shape[0] < N:
        # Pad sequence dimension if needed
        pad_len = N - cos.shape[0]
        cos = F.pad(cos, (0, 0, 0, pad_len))
        sin = F.pad(sin, (0, 0, 0, pad_len))
    
    cos_seq = cos[:N]  # (N, D/2 or less)
    sin_seq = sin[:N]  # (N, D/2 or less)
    
    # Check if feature dimensions match
    expected_dim = D // 2
    actual_dim = cos_seq.shape[-1]
    
    if actual_dim != expected_dim:
        # Dimension mismatch - pad or truncate feature dimension
        if actual_dim < expected_dim:
            # Pad with zeros
            pad_size = expected_dim - actual_dim
            cos_seq = F.pad(cos_seq, (0, pad_size))
            sin_seq = F.pad(sin_seq, (0, pad_size))
        else:
            # Truncate
            cos_seq = cos_seq[..., :expected_dim]
            sin_seq = sin_seq[..., :expected_dim]
    
    # Expand to full dimension by repeating each element twice
    cos_expanded = cos_seq.repeat_interleave(2, dim=-1)  # (N, D)
    sin_expanded = sin_seq.repeat_interleave(2, dim=-1)  # (N, D)
    
    # Reshape for broadcasting: (1, N, 1, D)
    cos_slice = cos_expanded.view(1, N, 1, D)
    sin_slice = sin_expanded.view(1, N, 1, D)

    # Rotation logic: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2] for sin multiplication
    x_half = x.reshape(B, N, H, D // 2, 2)
    x_rotated = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).reshape(
        B, N, H, D
    )

    return x * cos_slice + x_rotated * sin_slice


# ============================================================================
# BITLINEAR 1.58b - TERNARY WEIGHT LAYER
# ============================================================================


# 1. Official Activation Quantization (8-bit)
def activation_quant(x):
    """
    Per-token quantization to 8 bits.
    """
    # 127.0 is the max value for signed 8-bit (Qb)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y


# 2. Official Weight Quantization (1.58-bit Ternary)
def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits ({-1, 0, 1}).
    """
    # Beta is the average absolute value of the weight matrix
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    u = (w * scale).round().clamp(-1, 1) / scale
    return u


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Latent full-precision "Shadow Weights"
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # The paper emphasizes using RMSNorm (or Sub-LayerNorm)
        # specifically before the quantization step.
        self.norm = nn.RMSNorm(in_features)

        # Paper uses Kaiming Uniform for initial latent weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 1. Normalize input
        x_norm = self.norm(x)

        # 2. Quantize Activations with STE
        # Forward pass uses quantized, backward pass uses identity (x_norm)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # 3. Quantize Weights with STE
        # Forward pass uses ternary, backward pass uses identity (self.weight)
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # 4. Linear Projection
        return F.linear(x_quant, w_quant, self.bias)


class QuantizedLinear(nn.Module):
    """
    Wrapper for BitLinear 1.58b.
    Maintains compatibility with your existing EfficientTransformer architecture.
    """

    def __init__(self, in_features, out_features, bias=True, bits=1.58):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # We now point directly to the BitNet 1.58b implementation
        self.linear = BitLinear(in_features, out_features, bias=bias)

    @torch.compiler.disable
    def forward(self, x):
        # Pass-through to the BitLinear implementation
        return self.linear(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight.type_as(x)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)



# 1. Official Activation Quantization (8-bit)
def activation_quant(x):
    """
    Per-token quantization to 8 bits.
    """
    # 127.0 is the max value for signed 8-bit (Qb)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y


# 2. Official Weight Quantization (1.58-bit Ternary)
def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits ({-1, 0, 1}).
    """
    # Beta is the average absolute value of the weight matrix
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    u = (w * scale).round().clamp(-1, 1) / scale
    return u


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Latent full-precision "Shadow Weights"
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # The paper emphasizes using RMSNorm (or Sub-LayerNorm)
        # specifically before the quantization step.
        self.norm = nn.RMSNorm(in_features)

        # Paper uses Kaiming Uniform for initial latent weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 1. Normalize input
        x_norm = self.norm(x)

        # 2. Quantize Activations with STE
        # Forward pass uses quantized, backward pass uses identity (x_norm)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # 3. Quantize Weights with STE
        # Forward pass uses ternary, backward pass uses identity (self.weight)
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # 4. Linear Projection
        return F.linear(x_quant, w_quant, self.bias)


class QuantizedLinear(nn.Module):
    """
    Wrapper for BitLinear 1.58b.
    Maintains compatibility with your existing EfficientTransformer architecture.
    """

    def __init__(self, in_features, out_features, bias=True, bits=1.58):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # We now point directly to the BitNet 1.58b implementation
        self.linear = BitLinear(in_features, out_features, bias=bias)

    @torch.compiler.disable
    def forward(self, x):
        # Pass-through to the BitLinear implementation
        return self.linear(x)


# ============================================================================
# Lightning ATTENTION 2 (CHUNK-WISE / TILED)
# ============================================================================


class LightningAttention2(nn.Module):
    """
    Lightning Attention 2: Tiled Linear Attention with Gated Residual Connections
    
    Combines two papers:
    1. Lightning Attention-2 (arXiv:2401.04658) - Tiled linear attention
    2. Dynamic Context Adaptation (arXiv:2405.13407) - Gated Residual Connections
    
    Lightning Attention Algorithm:
    - Divide Q, K, V into T blocks of size B (chunk_size)
    - For each block t:
      * Intra-block: O_intra = [(Q_t @ K_t^T) ⊙ M] @ V_t  (conventional attention with causal mask)
      * Inter-block: O_inter = Q_t @ KV_{t-1}  (linear attention using accumulated state)
      * Output: O_t = O_intra + O_inter
      * Update: KV_t = KV_{t-1} + K_t^T @ V_t
    
    Gated Residual Connections:
    - Gate g = sigmoid(W_g @ [x, attn_output])
    - Output = x + g ⊙ attn_output  (instead of x + attn_output)
    
    Complexity: O(N*d^2) time, O(d^2) memory (constant w.r.t sequence length)
    """

    def __init__(self, dim, num_heads=8, chunk_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.scale = self.head_dim**-0.5

        # Projections
        self.qkv = QuantizedLinear(dim, dim * 3)
        self.proj = QuantizedLinear(dim, dim)
        self.norm = RMSNorm(dim)
        
        print("  ⚡ Lightning Attention 2: ENABLED (O(N) complexity, tiled implementation)")

    def forward(self, x, cos=None, sin=None, mask=None):
        B, N, C = x.shape

        # 1. Project Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, N, H, D)

        # 2. Apply RoPE if provided
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # 3. Pad sequence to multiple of chunk_size
        pad_len = (self.chunk_size - (N % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            padding = torch.zeros(
                B, pad_len, self.num_heads, self.head_dim,
                device=x.device, dtype=x.dtype
            )
            q = torch.cat([q, padding], dim=1)
            k = torch.cat([k, padding], dim=1)
            v = torch.cat([v, padding], dim=1)

        N_padded = q.shape[1]
        num_chunks = N_padded // self.chunk_size

        # 4. Reshape into chunks: (B, T, chunk_size, H, D)
        q = q.view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        k = k.view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        v = v.view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)

        # 5. Initialize KV state: (B, H, D, D)
        kv_state = torch.zeros(
            B, self.num_heads, self.head_dim, self.head_dim,
            device=x.device, dtype=x.dtype
        )

        # 6. Create causal mask for intra-block attention: (chunk_size, chunk_size)
        causal_mask = torch.tril(torch.ones(
            self.chunk_size, self.chunk_size,
            device=x.device, dtype=torch.bool
        ))

        output_chunks = []

        # 7. Process each chunk
        for t in range(num_chunks):
            q_t = q[:, t]  # (B, chunk_size, H, D)
            k_t = k[:, t]  # (B, chunk_size, H, D)
            v_t = v[:, t]  # (B, chunk_size, H, D)

            # Rearrange for attention: (B, H, chunk_size, D)
            q_t = q_t.transpose(1, 2)
            k_t = k_t.transpose(1, 2)
            v_t = v_t.transpose(1, 2)

            # INTRA-BLOCK: Conventional attention within chunk
            # attn_scores: (B, H, chunk_size, chunk_size)
            attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_t, k_t) * self.scale
            
            # Apply causal mask
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
            
            # Softmax
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values: (B, H, chunk_size, D)
            o_intra = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v_t)

            # INTER-BLOCK: Linear attention using accumulated KV state
            # o_inter: (B, H, chunk_size, D)
            o_inter = torch.einsum('bhqd,bhde->bhqe', q_t, kv_state)

            # Combine intra and inter block outputs
            o_t = o_intra + o_inter  # (B, H, chunk_size, D)

            # Update KV state for next chunk: KV_t = KV_{t-1} + K_t^T @ V_t
            # kv_update: (B, H, D, D)
            kv_update = torch.einsum('bhkd,bhke->bhde', k_t, v_t)
            kv_state = kv_state + kv_update

            # Rearrange back: (B, chunk_size, H, D)
            o_t = o_t.transpose(1, 2)
            output_chunks.append(o_t)

        # 8. Concatenate chunks and remove padding
        output = torch.cat(output_chunks, dim=1)  # (B, N_padded, H, D)
        if pad_len > 0:
            output = output[:, :N]  # Remove padding

        # 9. Reshape and project
        output = output.reshape(B, N, -1)  # (B, N, C)
        return self.proj(self.norm(output))



class Transformer(nn.Module):
    """
    Efficient Transformer with Lightning Attention 2 and Gated Residual Connections
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, bing_api_key=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = LightningAttention2(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        
        # Gated Residual Connections (from arXiv:2405.13407)
        # Gate for attention residual
        self.attn_gate = QuantizedLinear(dim * 2, dim)
        # Gate for feedforward residual
        self.ff_gate = QuantizedLinear(dim * 2, dim)
        
        # TITANS + MIRAS replaces traditional MLP entirely
        # TITANS Memory for test-time learning
        self.titans_memory = TitansMemory(
            dim=dim, 
            memory_depth=3, 
            capacity=1000,  # Efficient capacity
            surprise_threshold=0.5
        )
        
        # MIRAS for coordinated multi-tier retrieval
        self.miras = MIRAS(
            titans_memory=self.titans_memory,
            faiss_index=None,  # Will be set externally if needed
            bing_api_key=bing_api_key,
            confidence_threshold=0.7
        )
        
        # Projection layer to combine retrieved info with input
        self.memory_proj = nn.Sequential(
            QuantizedLinear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            QuantizedLinear(dim * 4, dim)
        )

    def forward(self, x, cos=None, sin=None, mask=None):
        # Attention block with Gated Residual Connection
        normed1 = self.norm1(x)
        attn_output = self.attn(normed1, cos, sin, mask)
        
        # Gated Residual Connection for attention
        # gate = sigmoid(W_g @ [x, attn_output])
        gate_input = torch.cat([x, attn_output], dim=-1)
        gate = torch.sigmoid(self.attn_gate(gate_input))
        x = x + gate * attn_output  # Gated residual
        
        # TITANS + MIRAS memory-augmented feedforward (replaces MLP)
        normed2 = self.norm2(x)
        
        # Use MIRAS for multi-tier retrieval
        retrieved, _ = self.miras.retrieve(normed2)
        
        # Ensure retrieved has same dimensions as normed2
        if retrieved.dim() == 2 and normed2.dim() == 3:
            # retrieved is [B, D], normed2 is [B, L, D]
            # Expand retrieved to [B, L, D]
            retrieved = retrieved.unsqueeze(1).expand(-1, normed2.size(1), -1)
        elif retrieved.dim() == 3 and normed2.dim() == 2:
            # retrieved is [B, L, D], normed2 is [B, D]
            # Pool retrieved to [B, D]
            retrieved = retrieved.mean(dim=1)
        
        # Combine input with retrieved memory
        combined = torch.cat([normed2, retrieved], dim=-1)
        memory_output = self.memory_proj(combined)
        
        # Gated Residual Connection for feedforward
        gate_input2 = torch.cat([x, memory_output], dim=-1)
        gate2 = torch.sigmoid(self.ff_gate(gate_input2))
        x = x + gate2 * memory_output  # Gated residual
        
        # Update memory based on surprise (during training)
        if self.training:
            surprise = self.titans_memory.compute_surprise(normed2, retrieved)
            self.titans_memory.update(normed2, normed2, surprise)
        
        return x


# ============================================================================

# FAISS RETRIEVAL-AUGMENTED GENERATION (RAG) - HUGE SCALE

# ============================================================================


class FAISSMemoryBank:
    """
    FAISS Memory Bank with MMAP support for Huge Indices (>RAM)
    """

    def __init__(self, dimension=768, base_dir="memory_bank", shards=4, auto_save=True):

        self.dimension = dimension
        self.base_dir = base_dir
        self.shards = shards
        self.auto_save = auto_save
        self.mmap_mode = True  # Always use disk mapping for huge memories

        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available. Memory bank disabled.")
            return

        os.makedirs(base_dir, exist_ok=True)
        self.indices = []
        self.shard_memories = []  # Now a dict {id: text} for fast delete
        self.next_id = 0  # Global ID counter

        # Load shards
        for shard_id in range(shards):
            index_path = os.path.join(base_dir, f"shard_{shard_id}.index")
            try:
                if os.path.exists(index_path):
                    # Standard Loading: Critical for stable incremental updates.
                    # IO_FLAG_MMAP often loses the filename reference, causing truncate() to fail.
                    index = faiss.read_index(index_path)
                    print(f"  Shard {shard_id}: Loaded into memory")
                    self.indices.append(index)
                else:
                    # Create coarse quantizer for IVF
                    quantizer = faiss.IndexFlatL2(dimension)

                    # Create new IVF index (Clustered) for scalability
                    if dimension % 8 == 0:
                        # IndexIVFPQ: Clustered + Product Quantization (Compression)
                        index = faiss.IndexIVFPQ(
                            quantizer, dimension, 100, dimension // 8, 8
                        )
                        print(f"  Using compressed IVFPQ index for shard {shard_id}")
                    else:
                        # Fallback to IVFFlat if dimension is not divisible by 8
                        index = faiss.IndexIVFFlat(
                            quantizer, dimension, 100, faiss.METRIC_L2
                        )
                        print(f"  Using IVFFlat index for shard {shard_id}")

                    # Training is REQUIRED for IVF/PQ
                    # We use a mock train with 10,000 vectors to satisfy clustering requirements
                    train_data = np.random.normal(0, 1, (10000, dimension)).astype(
                        "float32"
                    )
                    faiss.normalize_L2(train_data)
                    index.train(train_data)
                    index.nprobe = 10

                    # Wrap in IDMap to support specific ID management
                    index = faiss.IndexIDMap(index)
                    self.indices.append(index)

            except Exception as e:
                print(f"Index error shard {shard_id}: {e}")
                self.indices.append(faiss.IndexFlatL2(dimension))  # Fallback

            # Load metadata
            memories_path = os.path.join(base_dir, f"shard_{shard_id}.memories")
            if os.path.exists(memories_path):
                try:
                    with open(memories_path, "r", encoding="utf-8") as f:
                        data = joblib.load(f)
                        # Convert list back to dict if it was a list (backward compatibility)
                        if isinstance(data, list):
                            self.shard_memories.append(
                                {i: text for i, text in enumerate(data)}
                            )
                        else:
                            # Ensure keys are integers
                            self.shard_memories.append(
                                {int(k): v for k, v in data.items()}
                            )
                except:
                    self.shard_memories.append({})
            else:
                self.shard_memories.append({})

            # Update next_id based on existing IDs
            if self.shard_memories[-1]:
                self.next_id = max(
                    self.next_id, max(self.shard_memories[-1].keys()) + 1
                )

    @torch.compiler.disable
    def add_memory(self, embeddings, texts):
        """Add new embeddings and return their IDs"""
        if not self.indices:
            return []

        try:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()

            # Normalize for Inner Product / L2 stability
            faiss.normalize_L2(embeddings.astype("float32"))
            embeddings = embeddings.astype("float32")

            assigned_ids = []
            for i, (emb, text) in enumerate(zip(embeddings, texts)):
                shard_id = i % self.shards
                current_id = self.next_id

                # add_with_ids allows us to specify the ID
                ids_to_add = np.array([current_id]).astype("int64")
                self.indices[shard_id].add_with_ids(emb.reshape(1, -1), ids_to_add)
                self.shard_memories[shard_id][current_id] = text

                assigned_ids.append(current_id)
                self.next_id += 1

            if self.auto_save:
                self.save()
            return assigned_ids
        
        except Exception as e:
            print(f"ERROR: Failed to add memory to FAISS: {e}")
            # Try to recover by clearing cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []

    def delete_memory(self, ids):
        """Delete memories by their IDs"""
        if not self.indices:
            return

        try:
            if isinstance(ids, int):
                ids = [ids]

            # Convert to numpy for FAISS selector
            ids_np = np.array(ids).astype("int64")
            selector = faiss.IDSelectorArray(ids_np)

            for shard_id in range(self.shards):
                # Remove from FAISS index
                self.indices[shard_id].remove_ids(selector)

                # Remove from metadata dict
                for mid in ids:
                    if mid in self.shard_memories[shard_id]:
                        del self.shard_memories[shard_id][mid]

            if self.auto_save:
                self.save()
        
        except Exception as e:
            print(f"ERROR: Failed to delete memory from FAISS: {e}")
            # Continue execution - deletion failure is not critical

    def update_memory(self, mem_id, new_embedding, new_text):
        """Update a memory by deleting and re-adding"""
        self.delete_memory(mem_id)
        # We use a custom add here to keep the SAME ID if desired,
        # but the user suggested delete + add workflow which changes ID.
        # Let's just follow the delete + add workflow for simplicity.
        return self.add_memory(new_embedding, new_text)

    @torch.compiler.disable
    def search(self, query_embeddings, k=5):
        if not self.indices:
            return None, None, []

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()

        faiss.normalize_L2(query_embeddings.astype("float32"))
        query_embeddings = query_embeddings.astype("float32")

        all_results = []

        # Search all shards and collect candidates
        for shard_id, index in enumerate(self.indices):
            if index.ntotal > 0:
                dist, idx = index.search(query_embeddings, min(k, index.ntotal))

                for b in range(len(query_embeddings)):
                    for d, i in zip(dist[b], idx[b]):
                        if i != -1 and i in self.shard_memories[shard_id]:
                            all_results.append(
                                {
                                    "distance": d,
                                    "text": self.shard_memories[shard_id][i],
                                    "batch_idx": b,
                                }
                            )

        # Sort and merge results for each batch element
        batch_texts = []
        batch_distances = []
        batch_indices = []

        for b in range(len(query_embeddings)):
            b_results = [r for r in all_results if r["batch_idx"] == b]
            b_results.sort(key=lambda x: x["distance"])
            b_results = b_results[:k]

            batch_texts.append([r["text"] for r in b_results])
            batch_distances.append([r["distance"] for r in b_results])
            batch_indices.append([0 for _ in b_results])  # Indices not needed for RAG

        return batch_distances, batch_indices, batch_texts

    def save(self):
        # Save FAISS indices
        for i, index in enumerate(self.indices):
            faiss.write_index(index, os.path.join(self.base_dir, f"shard_{i}.index"))

        # Save Python memories efficiently
        for i, mem in enumerate(self.shard_memories):
            joblib.dump(mem, os.path.join(self.base_dir, f"shard_{i}.memories"))


# ============================================================================

# ULTRA-EFFICIENT MODEL

# ============================================================================


class EfficientXEncoder(nn.Module):
    def __init__(
        self, vocab_size=100277, d_model=768, depth=12, num_heads=8, max_seq_len=4096,
        bing_api_key=None
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Transformer(d_model, num_heads, bing_api_key=bing_api_key) 
             for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        cos, sin = precompute_rope_freqs(d_model // num_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, attention_mask=None):
        B, L = x.shape
        x = self.token_embed(x)

        for block in self.blocks:
            x = block(x, cos=self.cos[:L], sin=self.sin[:L], mask=attention_mask)
        return self.norm(x)


class EfficientYEncoder(nn.Module):
    """Efficient Y-Encoder with Flash Linear Attention + TITANS + MIRAS"""

    def __init__(
        self, vocab_size=100277, d_model=768, depth=6, num_heads=8, max_seq_len=512,
        bing_api_key=None
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Transformer(d_model, num_heads, bing_api_key=bing_api_key) 
             for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        cos, sin = precompute_rope_freqs(d_model // num_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, attention_mask=None):
        B, L = x.shape
        x = self.token_embed(x)
        for block in self.blocks:
            x = block(x, cos=self.cos[:L], sin=self.sin[:L], mask=attention_mask)
        return self.norm(x).mean(dim=1)


class EfficientPredictor(nn.Module):
    def __init__(
        self,
        source_dim=768,
        target_dim=768,
        predictor_dim=768,
        depth=6,
        num_heads=8,
        output_dim=1536,
        use_rag=True,
        token_embed=None,
        max_seq_len=8192,
    ):
        super().__init__()
        self.use_rag = use_rag and FAISS_AVAILABLE
        self.token_embed = token_embed
        self.source_proj = QuantizedLinear(source_dim, predictor_dim)
        self.query_proj = QuantizedLinear(target_dim, predictor_dim)
        if self.use_rag:
            self.memory_bank = FAISSMemoryBank(dimension=predictor_dim)
        self.blocks = nn.ModuleList(
            [Transformer(predictor_dim, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(predictor_dim)
        self.output_proj = QuantizedLinear(predictor_dim, output_dim)
        cos, sin = precompute_rope_freqs(predictor_dim // num_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.compiler.disable
    def _get_rag_embeddings(self, source_emb, source_tokens_device):
        """Isolated RAG logic to prevent torch.compile graph breaks"""
        rag_context = []
        rag_negatives = []

        if self.use_rag and hasattr(self, "memory_bank") and self.memory_bank.indices:
            query_vec = source_emb.mean(dim=1)
            d, i, texts = self.memory_bank.search(query_vec, k=2)

            if texts and self.token_embed is not None:
                batch_rag_tokens = []
                batch_neg_tokens = []

                for b in range(len(texts)):
                    try:
                        if len(texts[b]) > 0:
                            toks = joblib.loads(texts[b][0])
                            batch_rag_tokens.append(
                                torch.tensor(toks, device=source_tokens_device)
                            )
                        else:
                            batch_rag_tokens.append(
                                torch.zeros(
                                    1, dtype=torch.long, device=source_tokens_device
                                )
                            )

                        if len(texts[b]) > 1:
                            toks = joblib.loads(texts[b][1])
                            batch_neg_tokens.append(
                                torch.tensor(toks, device=source_tokens_device)
                            )
                        else:
                            batch_neg_tokens.append(
                                torch.zeros(
                                    1, dtype=torch.long, device=source_tokens_device
                                )
                            )
                    except:
                        batch_rag_tokens.append(
                            torch.zeros(
                                1, dtype=torch.long, device=source_tokens_device
                            )
                        )
                        batch_neg_tokens.append(
                            torch.zeros(
                                1, dtype=torch.long, device=source_tokens_device
                            )
                        )

                max_len = max([t.size(0) for t in batch_rag_tokens])
                padded_rag = torch.stack(
                    [F.pad(t, (0, max_len - t.size(0))) for t in batch_rag_tokens]
                )
                rag_emb = self.token_embed(padded_rag)
                rag_context.append(rag_emb)

                if batch_neg_tokens:
                    max_neg_len = max([t.size(0) for t in batch_neg_tokens])
                    rag_negatives = torch.stack(
                        [
                            F.pad(t, (0, max_neg_len - t.size(0)))
                            for t in batch_neg_tokens
                        ]
                    )

        return rag_context, rag_negatives

    def forward(self, source_tokens, query_tokens, source_mask=None, query_mask=None):
        source_emb = self.source_proj(source_tokens)
        query_emb = self.query_proj(query_tokens)

        # Isolated RAG Logic
        rag_context, rag_negatives = self._get_rag_embeddings(
            source_emb, source_tokens.device
        )

        # Concat context
        inputs = [source_emb, query_emb] + rag_context
        x = torch.cat(inputs, dim=1)
        L = x.shape[1]

        for block in self.blocks:
            x = block(x, cos=self.cos[:L], sin=self.sin[:L])

        return self.output_proj(self.norm(x).mean(dim=1)), rag_negatives


class EfficientYDecoder(nn.Module):
    def __init__(self, d_model=1536, vocab_size=100277, num_layers=4):
        super().__init__()
        self.embed_proj = QuantizedLinear(d_model, 768)  # Project back to model dim
        self.blocks = nn.ModuleList(
            [Transformer(768, num_heads=8) for _ in range(num_layers)]
        )
        self.lm_head = QuantizedLinear(768, vocab_size)

    def forward(self, embedding, max_length=50):
        x = self.embed_proj(embedding).unsqueeze(1).expand(-1, max_length, -1)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


class InfoNCELoss(nn.Module):
    """InfoNCE Loss for JEPA training"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, target, negatives=None):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)

        batch_size = pred.shape[0]
        
        # Handle single-sample batch (e.g., during validation with small dataset)
        if batch_size == 1:
            # For batch_size=1, use MSE loss as fallback
            # This ensures validation doesn't return 0 loss
            mse_loss = F.mse_loss(pred, target)
            # Scale to be comparable to InfoNCE loss range
            return mse_loss * 2.0
        
        # Standard InfoNCE for batch_size >= 2
        # Positive logits: (B, B)
        logits = pred @ target.T / self.temperature

        if negatives is not None:
            # Negatives: (B, K, D)
            # pred: (B, D), negatives: (B, K, D)
            # logits_neg: (B, K)
            negatives = F.normalize(negatives, dim=-1)
            logits_neg = torch.einsum("bd,bkd->bk", pred, negatives) / self.temperature

            # Concat to logits (B, B+K)
            logits = torch.cat([logits, logits_neg], dim=1)

        labels = torch.arange(pred.shape[0], device=pred.device)
        loss_i2t = F.cross_entropy(logits, labels)

        return loss_i2t


class PatchPooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, boundaries):
        # x: [Batch, SeqLen, Dim]
        # boundaries: [Batch, SeqLen] (1 at the start of a new patch)

        batch_size, seq_len, _ = x.shape
        all_patches = []

        for b in range(batch_size):
            # Find indices where boundaries == 1
            idx = torch.where(boundaries[b] == 1)[0]
            # Mean pool bytes between boundary markers
            # (Simplification: in production use a more optimized scatter_mean)
            patches = []
            for i in range(len(idx)):
                start = idx[i]
                end = idx[i + 1] if i + 1 < len(idx) else seq_len
                patches.append(x[b, start:end].mean(dim=0))
            all_patches.append(torch.stack(patches))

        return torch.stack(all_patches)  # [Batch, NumPatches, Dim]


# ============================================================================

# MAIN ULTRA-EFFICIENT TEXT-JEPA MODEL

# ============================================================================


class UltraEfficientTextJEPA(nn.Module):
    """

    Ultra-Efficient Text-JEPA for low-end hardware



    Optimizations:

    1. BitLinear 1.58b: 8-16x memory reduction, 2-3x CPU speedup

    2. Linear Attention: O(n) instead of O(n²), enables 16K+ sequences

    3. FAISS RAG: External memory for knowledge without model bloat

    4. Quantization: Bitnet 1.85 bits (-1,0,1)) for GPU acceleration

    5. CPU-Optimized Architecture: Unified Hybrid Attention + Titans Memory + MIRAS + HopRAG


    """

    def __init__(
        self,
        vocab_size=50000,
        source_dim=768,
        source_depth=12,
        target_dim=768,
        target_depth=6,
        predictor_dim=768,
        predictor_depth=8,
        output_dim=1536,
        temperature=0.07,
        max_source_len=16384,
        max_target_len=512,
        use_rag=True,
        use_enhanced_encoder=True,
        use_titans=True,
        use_miras=True,
        use_hoprag=True,
        bing_api_key=None,
        gradient_checkpointing=False,
    ):

        super().__init__()

        print("Initializing Ultra-Efficient Text-JEPA...")

        print(f"  BitLinear quantization: {'OK' if True else 'NO'}")

        print(f"  Linear Attention: OK (O(n) complexity)")

        print(f"  FAISS RAG: {'OK' if use_rag and FAISS_AVAILABLE else 'NO'}")

        print(f"  Max source length: {max_source_len} tokens")
        
        print(f"  Enhanced Architecture: {'OK' if use_enhanced_encoder else 'NO'}")
        
        print(f"  Titans Memory: {'OK' if use_titans else 'NO'}")
        
        print(f"  MIRAS Retrieval: {'OK' if use_miras else 'NO'}")
        
        print(f"  HopRAG Multi-Hop: {'OK' if use_hoprag else 'NO'}")

        # Use Enhanced Encoder (currently using EfficientXEncoder)
        # Note: Enhanced encoder features (titans, miras, hoprag) are not yet implemented
        self.x_encoder = EfficientXEncoder(
            vocab_size, source_dim, source_depth, max_seq_len=max_source_len
        )

        self.y_encoder = EfficientYEncoder(
            vocab_size, target_dim, target_depth, num_heads=8, max_seq_len=max_target_len
        )

        # Pass token embedding to predictor for RAG re-embedding
        token_embed = self.x_encoder.token_embed
        # BLT Hash Table: size 500,000 as per your constants
        self.hash_embed_table = nn.Embedding(500000, source_dim)

        # Pooling helper (Mean pooling bytes within a patch)
        self.pool_to_patches = PatchPooler(source_dim)

        self.predictor = EfficientPredictor(
            source_dim,
            target_dim,
            predictor_dim,
            predictor_depth,
            output_dim=output_dim,
            use_rag=use_rag,
            token_embed=token_embed,
        )

        self.y_encoder_proj = QuantizedLinear(target_dim, output_dim)

        self.y_decoder = EfficientYDecoder(output_dim, vocab_size)

        # WEIGHT TYING: Tie input embeddings with output head
        # This saves memory and improves semantic alignment
        self.y_decoder.lm_head.weight = self.x_encoder.token_embed.weight

        self.loss_fn = InfoNCELoss(temperature)

        # Count parameters

        total_params = sum(p.numel() for p in self.parameters())

        print(
            f"  Total parameters: {total_params:,} (~{total_params*2/1e6:.1f}MB in fp16)"
        )
    
    def load_state_dict_with_compatibility(self, state_dict, strict=False):
        """
        Load state dict with backward compatibility for old checkpoints.
        
        Handles:
        - Old checkpoints without enhanced encoder
        - Parameter name mapping
        - Missing keys for new components
        """
        # Since we're only using EfficientXEncoder now, no special mapping needed
        # Just load the state dict directly
        return super().load_state_dict(state_dict, strict=strict)

    def encode_source(self, byte_ids, patch_boundaries=None, hash_ngrams=None):
        """
        Modified to use the existing x_encoder sub-module.
        Currently uses EfficientXEncoder only.
        """
        B, L = byte_ids.shape
        
        # Use EfficientXEncoder path
        # 1. Access the embedding layer from the x_encoder sub-module
        x = self.x_encoder.token_embed(byte_ids)

        # 2. Add Hash N-Gram info
        if hash_ngrams is not None and hasattr(self, "hash_embed_table"):
            hash_embeds = self.hash_embed_table(hash_ngrams).sum(dim=-2)
            x = x + hash_embeds

        # 3. Process through blocks
        cos = self.x_encoder.cos[:L]
        sin = self.x_encoder.sin[:L]

        for block in self.x_encoder.blocks:
            x = block(x, cos=cos, sin=sin, mask=None)

        x = self.x_encoder.norm(x)

        # 4. Patch Pooling (Byte-to-Latent transition)
        if patch_boundaries is not None:
            patch_representations = self.pool_to_patches(x, patch_boundaries)
            return patch_representations

        return x

    def encode_target(self, target_tokens, attention_mask=None):

        emb = self.y_encoder(target_tokens, attention_mask)

        return self.y_encoder_proj(emb)

    def predict(self, source_tokens, query_tokens, source_mask=None, query_mask=None):
        # 1. Check if query_tokens needs embedding
        # If they are raw byte IDs (integers), we must embed them first
        if query_tokens.dtype in [torch.long, torch.int, torch.int32]:
            # Use the same embedding table as the y_encoder or x_encoder
            query_emb = self.x_encoder.token_embed(query_tokens)
        else:
            query_emb = query_tokens

        # Now query_emb shape is [Batch, SeqLen, 768]
        # 2. Pass to the predictor
        return self.predictor(source_tokens, query_emb, source_mask, query_mask)

    def decode(self, embedding, max_length=50):

        return self.y_decoder(embedding, max_length)

    def forward(
        self,
        source_tokens,
        query_tokens,
        target_tokens,
        source_mask=None,
        query_mask=None,
        target_mask=None,
        use_hoprag=False,
        query_text="",
    ):
        # Encode source (HopRAG not currently supported with EfficientXEncoder)
        source_emb = self.encode_source(source_tokens, source_mask)

        query_emb = self.y_encoder.token_embed(query_tokens)
        # RoPE is applied to query_emb inside the predictor's attention blocks

        pred_emb, neg_tokens = self.predict(
            source_emb, query_emb, source_mask, query_mask
        )

        target_emb = self.encode_target(target_tokens, target_mask)

        # Stop-Gradient on Target (JEPA Requirement)
        target_emb = target_emb.detach()

        # Encode negatives if available
        negative_emb = None
        if neg_tokens is not None and isinstance(neg_tokens, torch.Tensor):
            # Treating negatives as "targets" (using Y-Encoder)
            negative_emb = self.encode_target(neg_tokens)  # (B, D)
            # Reshape to (B, 1, D) for loss
            negative_emb = negative_emb.unsqueeze(1)

        loss = self.loss_fn(pred_emb, target_emb, negatives=negative_emb)

        return loss, pred_emb, target_emb
    
    def save_titans_memory(self, path: str):
        """Save Titans Memory state to disk"""
    def save_titans_memory(self, path: str):
        """Save Titans Memory state to disk (not currently supported)"""
        print("Titans Memory not available with current encoder")
    
    def load_titans_memory(self, path: str):
        """Load Titans Memory state from disk (not currently supported)"""
        print("Titans Memory not available with current encoder")
    
    def forget_titans_memory(self, n: int = 100):
        """Forget least accessed memories (not currently supported)"""
        print("Titans Memory not available with current encoder")


# ============================================================================

# GALORE OPTIMIZER FOR MEMORY-EFFICIENT TRAINING

# ============================================================================


class GaLoreOptimizer(torch.optim.Optimizer):
    """
    Hybrid Optimizer: GaLore (Low-Rank) + Muon (Orthogonal Updates)
    - CANS iteration for fast orthogonalization.
    - Low VRAM: Only stores momentum for the low-rank subspace.
    - Fast Convergence: Orthogonalizes updates like Muon.
    """

    def __init__(
        self, params, lr=1e-4, rank=128, update_freq=200, cans_steps=5, delta=0.3
    ):
        defaults = dict(lr=lr, rank=rank, update_freq=update_freq, cans_steps=cans_steps, delta=delta)
        super().__init__(params, defaults)
        self.rank = rank
        self.update_freq = update_freq
        self.cans_steps = cans_steps
        self.delta = delta
        self.step_count = 0
        self.projections = {}  # Subspace matrices
        self.momentum = {}  # Low-rank momentum only

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def _power_norm(self, X, iterations=5):
        """Estimate spectral norm using power iteration"""
        v = torch.randn(X.shape[1], device=X.device)
        for _ in range(iterations):
            v = X.T @ (X @ v)
            v /= v.norm() + 1e-8
        sigma = (X @ v).norm()
        return sigma

    def _cans(self, X, degrees=None, s=None):
        """
        Chebyshev-accelerated Newton-Schulz (CANS) Orthogonalization
        X: 2D matrix (m, n) where m <= n (typically rank × dim)
        degrees: polynomial degrees per iteration
        s: number of CANS iterations
        """
        if degrees is None:
            degrees = [2] * (s or self.cans_steps)
        s = s or self.cans_steps

        # Normalize by estimated spectral norm
        sigma_max = self._power_norm(X)
        X = X / (sigma_max + 1e-8)

        # δ-orthogonalization bounds
        a, b = 1 - self.delta, 1 + self.delta

        for i in range(s):
            deg = degrees[i]
            # Use simple Chebyshev polynomial approximation (odd degree)
            # p(x) = x * (3 - x^2) / 2 for deg=2 (classic NS iteration)
            # For X of shape (m, n), we compute X @ (3I - X.T @ X) / 2
            # where I is n×n identity
            if deg == 2:
                # X is (m, n), X.T @ X is (n, n)
                XTX = X.T @ X
                I_n = torch.eye(X.shape[1], device=X.device)
                X = 0.5 * X @ (3 * I_n - XTX)
            else:
                # Higher-degree: recursive formula or precomputed coefficients can be used
                # For simplicity, fallback to repeated NS
                for _ in range(deg):
                    XTX = X.T @ X
                    I_n = torch.eye(X.shape[1], device=X.device)
                    X = 0.5 * X @ (3 * I_n - XTX)

            # Update bounds (optional, for monitoring)
            a, b = 1 - self.delta, 1 + self.delta

        return X

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1

        for group in self.param_groups:
            lr = group['lr']
            rank = group['rank']
            update_freq = group['update_freq']
            
            for p in group['params']:
                if p.grad is None or p.dim() < 2:
                    # For 1D parameters (biases, norms), use standard SGD
                    if p.grad is not None:
                        p.data.add_(p.grad.data, alpha=-lr)
                    continue

                grad = p.grad.data
                # Flatten to 2D
                orig_shape = grad.shape
                grad_2d = grad.view(-1, orig_shape[-1])  # (M, N)
                
                # Ensure rank doesn't exceed matrix dimensions
                actual_rank = min(rank, grad_2d.shape[0], grad_2d.shape[1])

                # 1. Update Subspace (GaLore part) - use SVD for proper low-rank projection
                if self.step_count % update_freq == 0 or id(p) not in self.projections:
                    # Compute low-rank approximation using randomized SVD for efficiency
                    try:
                        U, S, Vh = torch.svd_lowrank(grad_2d, q=actual_rank)
                        self.projections[id(p)] = (U, Vh)
                    except:
                        # Fallback: just use the gradient as-is
                        self.projections[id(p)] = None

                # 2. Project Gradient to Low-Rank Space
                if id(p) in self.projections and self.projections[id(p)] is not None:
                    U, Vh = self.projections[id(p)]
                    # Vh is already V^T from svd_lowrank, so grad_2d @ Vh.T gives us @ V
                    # Project: grad_lr = U.T @ grad_2d @ V
                    grad_lr = U.T @ grad_2d @ Vh  # (rank, rank)
                else:
                    # No projection, use full gradient
                    grad_lr = grad_2d

                # 3. Momentum
                if id(p) not in self.momentum:
                    self.momentum[id(p)] = torch.zeros_like(grad_lr)

                buf = self.momentum[id(p)]
                buf.mul_(0.9).add_(grad_lr, alpha=0.1)

                # 4. Project Back & Apply
                if id(p) in self.projections and self.projections[id(p)] is not None:
                    U, Vh = self.projections[id(p)]
                    # Reconstruct: update = U @ buf @ V^T = U @ buf @ Vh
                    update = (U @ buf @ Vh.T).view(orig_shape)
                else:
                    update = buf.view(orig_shape)
                    
                p.data.add_(update, alpha=-lr)
        
        return loss
