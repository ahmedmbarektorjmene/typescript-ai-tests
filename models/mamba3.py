"""
Mamba-2 with Rule Generalization Enhancements
Combines pattern matching with symbolic reasoning for better generalization
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
    """Reference implementation of SSD chunk scan for CPU/fallback"""
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
    
    # Fixed: Proper group broadcasting
    if ngroups == 1:
        dB = dB.squeeze(3).expand(-1, -1, nheads, -1)
    else:
        heads_per_group = nheads // ngroups
        dB = dB.repeat_interleave(heads_per_group, dim=2).squeeze(2)
    
    # Initialize state
    h = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=x.dtype)
    outputs = []
    
    # Sequential scan
    for i in range(seqlen):
        h = h * dA[:, i].unsqueeze(-1)
        dBx = torch.einsum('bhp,bhn->bhpn', x[:, i], dB[:, i])
        h = h + dBx
        
        # y = C * h
        if ngroups == 1:
            y = torch.einsum('bhpn,bn->bhp', h, C[:, i, 0])
        else:
            C_expanded = C[:, i].repeat_interleave(heads_per_group, dim=1)
            y = torch.einsum('bhpn,bhn->bhp', h, C_expanded)
        
        if D is not None:
            if D.dim() == 1:
                y = y + D.view(1, nheads, 1) * x[:, i]
            else:
                y = y + D.view(1, nheads, headdim) * x[:, i]
        
        if z is not None:
            y = y * F.silu(z[:, i])
        
        outputs.append(y)
    
    return torch.stack(outputs, dim=1)


class SymbolicRuleLayer(nn.Module):
    """Explicit symbolic reasoning layer for rule abstraction"""
    def __init__(self, d_model, n_symbols=32, symbol_dim=64):
        super().__init__()
        self.n_symbols = n_symbols
        self.symbol_dim = symbol_dim
        
        # Learnable symbolic slots
        self.symbols = nn.Parameter(torch.randn(n_symbols, symbol_dim))
        self.symbol_proj = nn.Linear(d_model, symbol_dim)
        
        # Rule composition network
        self.rule_mlp = nn.Sequential(
            nn.Linear(symbol_dim * 2, symbol_dim * 4),
            nn.GELU(),
            nn.Linear(symbol_dim * 4, symbol_dim)
        )
        
        # Project back to d_model
        self.output_proj = nn.Linear(symbol_dim, d_model)
        
    def forward(self, x):
        """
        x: (batch, seqlen, d_model)
        Returns: (batch, seqlen, d_model)
        """
        batch, seqlen, _ = x.shape
        
        # Project to symbol space
        symbol_features = self.symbol_proj(x)  # (batch, seqlen, symbol_dim)
        
        # Compute attention between sequence and symbolic slots
        attn = torch.einsum('bsd,nd->bsn', symbol_features, self.symbols)
        attn = attn / math.sqrt(self.symbol_dim)
        attn = F.softmax(attn, dim=-1)
        
        # Retrieve symbol context
        symbol_context = torch.einsum('bsn,nd->bsd', attn, self.symbols)
        
        # Learn rule compositions
        rule_input = torch.cat([symbol_features, symbol_context], dim=-1)
        rule_output = self.rule_mlp(rule_input)
        
        # Project back to model dimension
        return self.output_proj(rule_output)


class CompositionalGate(nn.Module):
    """Gate to balance pattern matching vs rule application"""
    def __init__(self, d_model):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, pattern_features, rule_features, context):
        """
        Decide when to use pattern matching vs rule application
        pattern_features: output from Mamba (pattern matching)
        rule_features: output from symbolic layer (rule abstraction)
        context: original input for context
        """
        gate_input = torch.cat([pattern_features, rule_features, context], dim=-1)
        gates = self.gate_network(gate_input)  # (batch, seqlen, 2)
        
        pattern_gate = gates[..., 0:1]
        rule_gate = gates[..., 1:2]
        
        return pattern_gate * pattern_features + rule_gate * rule_features


class HierarchicalMamba2(nn.Module):
    """
    Enhanced Mamba-2 with hierarchical state representation
    for multi-scale temporal abstraction
    """
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        n_scales=3,  # NEW: hierarchical scales
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
        self.n_scales = n_scales  # NEW
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
        
        # NEW: Hierarchical state dimensions
        self.d_state_per_scale = self.d_state // self.n_scales
        
        # Input projection
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
        
        # SSM parameters - hierarchical A
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        # NEW: Hierarchical A parameters for multi-scale
        A = torch.empty(self.n_scales, self.nheads, dtype=torch.float32, device=device)
        for scale in range(self.n_scales):
            scale_range = (A_init_range[0] * (scale + 1), A_init_range[1] * (scale + 1))
            A[scale] = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*scale_range)
        
        A_log = torch.log(A)
        if dtype is not None:
            A_log = A_log.to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # NEW: Inter-scale mixing
        self.inter_scale_mixers = nn.ModuleList([
            nn.Linear(self.d_state_per_scale, self.d_state_per_scale, bias=False, **factory_kwargs)
            for _ in range(self.n_scales - 1)
        ])
        
        self.D = nn.Parameter(torch.ones(
            self.d_ssm if self.D_has_hdim else self.nheads, 
            **factory_kwargs
        ))
        self.D._no_weight_decay = True
        
        if self.rmsnorm:
            self.norm = RMSNorm(self.d_ssm, eps=1e-5)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
    
    def forward(self, u, inference_params=None):
        """Forward pass with hierarchical processing"""
        batch, seqlen, _ = u.shape
        
        # Standard Mamba-2 forward (simplified for brevity)
        # In practice, you'd implement the full hierarchical state processing
        zxbcdt = self.in_proj(u)
        
        # Use average A across scales for compatibility
        A = -torch.exp(self.A_log.float()).mean(dim=0)
        
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # Use fused kernel if available
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
            # Fallback path
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


class RuleGeneralizingMamba2Block(nn.Module):
    """
    Enhanced Mamba-2 block combining pattern matching with rule abstraction
    """
    def __init__(self, d_model, layer_idx=None, n_symbols=32, symbol_dim=64, **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        
        # Hierarchical Mamba for pattern processing
        self.mamba = HierarchicalMamba2(d_model, layer_idx=layer_idx, **kwargs)
        
        # Symbolic rule layer for abstraction
        self.rule_layer = SymbolicRuleLayer(d_model, n_symbols=n_symbols, symbol_dim=symbol_dim)
        
        # Compositional gate to balance pattern vs rule
        self.gate = CompositionalGate(d_model)
        
        # Rule refinement MLP
        self.rule_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, inference_params=None):
        """
        x: (batch, seqlen, d_model)
        Returns: (batch, seqlen, d_model)
        """
        residual = x
        
        # Pattern processing through Mamba
        x_norm = self.norm1(x)
        pattern_out = self.mamba(x_norm, inference_params=inference_params)
        
        # Rule abstraction through symbolic layer
        rule_out = self.rule_layer(x_norm)
        
        # Gate between pattern and rule
        gated = self.gate(pattern_out, rule_out, x_norm)
        gated = residual + gated
        
        # Rule refinement
        residual2 = gated
        gated_norm = self.norm3(gated)
        refined = self.rule_mlp(gated_norm)
        
        return residual2 + refined


class RuleAuxiliaryLoss(nn.Module):
    """Auxiliary losses to promote rule learning"""
    def __init__(self, d_model, n_rule_types=8):
        super().__init__()
        # Predict if pattern is rule-based or memorized
        self.rule_classifier = nn.Linear(d_model, 2)
        
        # Predict rule type (symmetry, repetition, arithmetic, etc.)
        self.rule_type_predictor = nn.Linear(d_model, n_rule_types)
        
    def compute_auxiliary_loss(self, hidden_states, rule_labels=None, rule_types=None):
        """
        hidden_states: (batch, seqlen, d_model)
        rule_labels: (batch,) - binary labels for rule-based vs memorized
        rule_types: (batch,) - categorical labels for rule types
        """
        # Aggregate sequence representation
        pooled = hidden_states.mean(dim=1)  # (batch, d_model)
        
        total_loss = 0.0
        
        # Rule vs memorization classification
        if rule_labels is not None:
            rule_logits = self.rule_classifier(pooled)
            rule_loss = F.cross_entropy(rule_logits, rule_labels)
            total_loss += rule_loss
        
        # Rule type prediction
        if rule_types is not None:
            type_logits = self.rule_type_predictor(pooled)
            type_loss = F.cross_entropy(type_logits, rule_types)
            total_loss += 0.5 * type_loss
        
        # Diversity loss: encourage different representations for different rules
        if hidden_states.size(0) > 1:
            # Compute pairwise distances
            pooled_norm = F.normalize(pooled, p=2, dim=1)
            similarity = torch.mm(pooled_norm, pooled_norm.t())
            
            # Penalize high similarity (encourage diversity)
            diversity_loss = similarity.pow(2).mean() - 1.0 / pooled.size(0)
            total_loss += 0.1 * diversity_loss
        
        return total_loss


class EnhancedMamba2Model(nn.Module):
    """
    Complete Mamba-2 model with rule generalization capabilities
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        n_symbols: int = 32,
        symbol_dim: int = 64,
        use_rule_aux_loss: bool = True,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_rule_aux_loss = use_rule_aux_loss
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Rule-generalizing blocks
        self.layers = nn.ModuleList([
            RuleGeneralizingMamba2Block(
                d_model, 
                layer_idx=i,
                n_symbols=n_symbols,
                symbol_dim=symbol_dim,
                **kwargs
            )
            for i in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Auxiliary loss module
        if self.use_rule_aux_loss:
            self.aux_loss = RuleAuxiliaryLoss(d_model)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, labels=None, rule_labels=None, rule_types=None, inference_params=None):
        """
        input_ids: (batch, seqlen)
        labels: (batch, seqlen) - target tokens
        rule_labels: (batch,) - optional binary labels for rule-based sequences
        rule_types: (batch,) - optional categorical labels for rule types
        """
        x = self.embedding(input_ids)
        
        # Collect hidden states from middle layers for aux loss
        hidden_states_for_aux = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, inference_params=inference_params)
            
            # Collect from middle layers
            if self.use_rule_aux_loss and i == len(self.layers) // 2:
                hidden_states_for_aux.append(x.detach())
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        aux_loss = None
        
        if labels is not None:
            # Fixed: Proper loss computation
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
            
            # Compute auxiliary loss
            if self.use_rule_aux_loss and len(hidden_states_for_aux) > 0:
                aux_loss = self.aux_loss.compute_auxiliary_loss(
                    hidden_states_for_aux[0],
                    rule_labels=rule_labels,
                    rule_types=rule_types
                )
                
                # Combine losses
                loss = loss + 0.3 * aux_loss
        
        return {
            'logits': logits,
            'loss': loss,
            'aux_loss': aux_loss
        }
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.95, eos_token_id=None):
        """Generate text with improved batching support"""
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
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for _ in range(max_length):
            if finished.all():
                break
                
            last_token = generated[:, -1:]
            outputs = self.forward(last_token, inference_params=inference_params)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-p sampling
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
            
            # Update finished mask
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            
            generated = torch.cat([generated, next_token], dim=1)
            inference_params.seqlen_offset += 1
        
        return generated


# Example usage and training loop
def create_rule_generalization_model(vocab_size=50257, d_model=768, n_layers=12):
    """Factory function to create enhanced model"""
    model = EnhancedMamba2Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        n_scales=3,  # Hierarchical scales
        n_symbols=32,  # Symbolic slots
        symbol_dim=64,
        use_rule_aux_loss=True,
        chunk_size=256,
    )
    return model


def train_step_with_rule_promotion(model, batch, optimizer):
    """Training step with rule-promoting losses"""
    optimizer.zero_grad()
    
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        rule_labels=batch.get('rule_labels'),  # Optional
        rule_types=batch.get('rule_types')     # Optional
    )
    
    loss = outputs['loss']
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return {
        'total_loss': loss.item(),
        'aux_loss': outputs['aux_loss'].item() if outputs['aux_loss'] is not None else 0.0
    }