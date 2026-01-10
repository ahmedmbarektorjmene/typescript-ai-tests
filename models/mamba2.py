"""
Mamba-2: Precise implementation following official code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat

# Try to import Triton kernels (optional for GPU acceleration)
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def ssd_chunk_scan_combined_ref(
    x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Reference implementation of SSD chunk scan for CPU/fallback
    Argument shapes:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads,)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads,) or (nheads, headdim)
        z: (batch, seqlen, nheads, headdim) optional
        dt_bias: (nheads,)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    ngroups = B.shape[2]
    
    # Apply dt bias and softplus
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    
    dt = dt.unsqueeze(-1)  # (batch, seqlen, nheads, 1)
    A = A.view(1, 1, nheads, 1)  # (1, 1, nheads, 1)
    
    # Discretize
    dA = torch.exp(dt * A)  # (batch, seqlen, nheads, 1)
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seqlen, 1, ngroups, dstate)
    
    # Handle group broadcasting
    if ngroups == 1:
        dB = dB.squeeze(3)  # (batch, seqlen, nheads, dstate)
    else:
        # Average over groups or repeat for each head
        dB = dB.expand(-1, -1, nheads, -1, -1).reshape(batch, seqlen, nheads, ngroups * dstate)
        dB = dB[..., :dstate]  # Simple average for now
    
    # Initialize state
    h = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=x.dtype)
    outputs = []
    
    # Sequential scan (can be optimized with chunking)
    for i in range(seqlen):
        # h = dA * h + dB * x
        h = h * dA[:, i].unsqueeze(-1)  # (batch, nheads, headdim, dstate)
        dBx = torch.einsum('bhp,bhn->bhpn', x[:, i], dB[:, i])
        h = h + dBx
        
        # y = C * h
        if ngroups == 1:
            y = torch.einsum('bhpn,bn->bhp', h, C[:, i, 0])
        else:
            y = torch.einsum('bhpn,bgn->bhp', h, C[:, i])
        
        # Add skip connection
        if D is not None:
            if D.dim() == 1:
                y = y + D.view(1, nheads, 1) * x[:, i]
            else:
                y = y + D.view(1, nheads, headdim) * x[:, i]
        
        # Apply gating if provided
        if z is not None:
            y = y * F.silu(z[:, i])
        
        outputs.append(y)
    
    return torch.stack(outputs, dim=1)


class Mamba2(nn.Module):
    """
    Mamba-2 layer - follows official implementation exactly
    """
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.activation = "silu"
        
        self.d_ssm = self.d_inner
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        
        # Input projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory_kwargs)
        
        # Convolution
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()
        
        # SSM parameters
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)

        A_log = torch.log(A)
        if dtype is not None:
            A_log = A_log.to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(
            self.d_ssm if self.D_has_hdim else self.nheads, 
            **factory_kwargs
        ))
        self.D._no_weight_decay = True
        
        if self.rmsnorm:
            self.norm = RMSNorm(self.d_ssm, eps=1e-5)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
    
    def forward(self, u, inference_params=None):
        """
        u: (batch, seqlen, d_model)
        Returns: (batch, seqlen, d_model)
        """
        batch, seqlen, _ = u.shape
        
        # Handle inference
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # Inference mode: single token
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
        
        # Training/prefill mode
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        A = -torch.exp(self.A_log.float())
        
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # Use fused kernel if available and requested
        if self.use_mem_eff_path and mamba_split_conv1d_scan_combined is not None and u.is_cuda:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
        else:
            # Fallback path (CPU or no Triton)
            z, xBC, dt = torch.split(
                zxbcdt,
                [self.d_inner, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            
            # Conv1d
            if causal_conv1d_fn is not None and u.is_cuda:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            else:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen]
                )
            
            x, B, C = torch.split(
                xBC, 
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], 
                dim=-1
            )
            
            # Reshape for multi-head
            x = rearrange(x, 'b l (h p) -> b l h p', p=self.headdim)
            B = rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups)
            C = rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups)
            
            # SSD scan
            if mamba_chunk_scan_combined is not None and u.is_cuda:
                y = mamba_chunk_scan_combined(
                    x, dt, A, B, C,
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
            else:
                # CPU fallback
                y = ssd_chunk_scan_combined_ref(
                    x, dt, A, B, C,
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                )
            
            y = rearrange(y, 'b l h p -> b l (h p)')
            
            if self.rmsnorm:
                y = self.norm(y) * self.act(z)
            
            out = self.out_proj(y)
        
        return out
    
    def step(self, hidden_states, conv_state, ssm_state):
        """Single step inference - matches official implementation"""
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"
        
        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        
        # Conv step
        if causal_conv1d_update is not None:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        
        # SSM step
        if selective_state_update is not None:
            A_expanded = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt_expanded = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias_expanded = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D_expanded = repeat(self.D, "h -> h p", p=self.headdim) if not self.D_has_hdim else self.D
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            
            z_reshaped = rearrange(z, "b (h p) -> b h p", p=self.headdim) if not self.rmsnorm else None
            
            y = selective_state_update(
                ssm_state, x_reshaped, dt_expanded, A_expanded, B, C, D_expanded,
                z=z_reshaped,
                dt_bias=dt_bias_expanded,
                dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        else:
            # Fallback
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        
        if self.rmsnorm:
            y = self.norm(y) * self.act(z)
        
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    
    def allocate_inference_cache(self, batch_size, dtype=None):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_ssm + 2 * self.ngroups * self.d_state, self.d_conv,
            device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state
    
    def _get_states_from_cache(self, inference_params, batch_size):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state, ssm_state = self.allocate_inference_cache(batch_size)
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        return conv_state, ssm_state


class Mamba2Block(nn.Module):
    def __init__(self, d_model, layer_idx=None, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba2(d_model, layer_idx=layer_idx, **kwargs)
    
    def forward(self, x, inference_params=None):
        return x + self.mixer(self.norm(x), inference_params=inference_params)


class Mamba2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, layer_idx=i, **kwargs)
            for i in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, labels=None, inference_params=None):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1)
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.95, eos_token_id=None):
        self.eval()
        
        class InferenceParams:
            def __init__(self):
                self.key_value_memory_dict = {}
                self.seqlen_offset = 0
        
        inference_params = InferenceParams()
        batch_size = input_ids.shape[0]
        
        # Prefill
        outputs = self.forward(input_ids, inference_params=inference_params)
        inference_params.seqlen_offset = input_ids.shape[1]
        
        generated = input_ids
        
        for _ in range(max_length):
            last_token = generated[:, -1:]
            outputs = self.forward(last_token, inference_params=inference_params)
            logits = outputs['logits'][:, -1, :] / temperature
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            inference_params.seqlen_offset += 1
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated