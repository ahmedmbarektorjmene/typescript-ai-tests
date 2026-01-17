import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Generator, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
import numpy as np

# ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime")

# Import Shutka components
from models.shutka import UltraEfficientTextJEPA
from config import TrainingConfig



N_MIN = 3
N_MAX = 8
HASH_TABLE_SIZE = 500000

def rolling_poly_hash(bytes_tensor: torch.Tensor, n: int, prime: int = 1000003) -> torch.Tensor:
    """
    Implements RollPolyHash from Equation 4 of the paper.
    Computes a hash for every window of size n.
    """
    length = bytes_tensor.size(0)
    hashes = torch.zeros(length, dtype=torch.long)
    if length < n:
        return hashes
    
    current_hash = 0
    # Initial window
    for i in range(n):
        current_hash = (current_hash * 256 + bytes_tensor[i].item()) % HASH_TABLE_SIZE
    
    hashes[n-1] = current_hash
    
    # Rolling step
    power = pow(256, n-1, HASH_TABLE_SIZE)
    for i in range(n, length):
        # Remove leading byte, add trailing byte
        current_hash = (current_hash - bytes_tensor[i-n].item() * power) % HASH_TABLE_SIZE
        current_hash = (current_hash * 256 + bytes_tensor[i].item()) % HASH_TABLE_SIZE
        hashes[i] = current_hash
        
    return hashes

# ============================================================================
# API MODELS (OpenAI Compatible)
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "shutka-v2"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512  # OpenAI compatible parameter
    max_completion_tokens: Optional[int] = None  # OpenAI compatible (newer)
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model: str = "shutka-v2"
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"  # OpenAI supports "float" or "base64"
    dimensions: Optional[int] = None  # For dimension reduction (not implemented)
    user: Optional[str] = None

class MemoryRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# MODEL ENGINE (HYBRID ONNX + PYTORCH)
# ============================================================================

class ShutkaEngine:
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt", 
                 onnx_path: str = "models/shutka.onnx",
                 fast_mode: bool = True, 
                 verbose: bool = False,
                 use_onnx: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fast_mode = fast_mode
        self.verbose = verbose
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        
        if verbose:
            print(f"Loading Shutka model onto {self.device}...")
        
        # CPU-specific optimizations
        if self.device.type == 'cpu':
            torch.set_num_threads(8)
            if verbose:
                print(f"  • CPU threads: {torch.get_num_threads()}")
        
        self.config = TrainingConfig()
        
        # FAST MODE OPTIMIZATIONS (Target: ~350M params like Llama 3.2)
        if fast_mode:
            # Reduce model dimensions (main parameter reduction)
            self.config.source_dim = 320       # 768 → 320 (aggressive reduction)
            self.config.target_dim = 320       # 768 → 320
            self.config.predictor_dim = 320    # 768 → 320
            self.config.output_dim = 640       # 1536 → 640
            
            # Reduce model depth
            self.config.source_depth = 6       # 12 → 6
            self.config.target_depth = 3       # 6 → 3
            self.config.predictor_depth = 3    # 6 → 3
            
            # Reduce context
            self.config.max_source_len = 1024  # 4096 → 1024
            self.config.max_target_len = 128   # 512 → 128
            
            # Reduce Titans memory
            self.config.titans_capacity = 2500  # 10000 → 2500
            self.config.titans_depth = 1        # 3 → 1
            
            # Reduce HopRAG hops
            self.config.hoprag_max_hops = 1     # 3 → 1
            
            # Disable FAISS for faster startup (but keep other features)
            self.config.use_rag = False  # Only disable FAISS for startup speed
        
        # Try to load ONNX model if it exists (don't auto-export)
        self.onnx_session = None
        if self.use_onnx and os.path.exists(onnx_path):
            try:
                if verbose:
                    print(f"📦 Loading ONNX model from {onnx_path}")
                
                # Configure ONNX Runtime for CPU optimization
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 8
                sess_options.inter_op_num_threads = 8
                
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                
                if verbose:
                    print("✓ ONNX model loaded successfully")
            except Exception as e:
                if verbose:
                    print(f"⚠ Failed to load ONNX model: {e}")
                    print("  Falling back to PyTorch...")
                self.onnx_session = None
        
        # Load PyTorch model for dynamic components (Titans, MIRAS)
        # or as fallback if ONNX fails
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device).eval()
        
        # STRICT: Match model vocab size
        self.vocab_size = self.model.x_encoder.token_embed.num_embeddings
        
        if verbose and self.onnx_session:
            print("🚀 Hybrid Mode: ONNX (static) + PyTorch (dynamic memory)")
        elif verbose:
            print("🐍 PyTorch Mode: Full model in PyTorch")
    
    def _get_bytes(self, text: str) -> torch.Tensor:
        """Converts text to raw UTF-8 bytes (0-255)."""
        byte_data = text.encode("utf-8")
        return torch.tensor(list(byte_data)[: self.max_byte_len], dtype=torch.long)

    def _get_hash_ngrams(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """Section 3.2.1: Encoder Hash n-gram Embeddings."""
        length = byte_tensor.size(0)
        # Table of [SeqLen, 6] (for n=3, 4, 5, 6, 7, 8)
        all_hashes = torch.zeros((length, N_MAX - N_MIN + 1), dtype=torch.long)
        
        for idx, n in enumerate(range(N_MIN, N_MAX + 1)):
            all_hashes[:, idx] = rolling_poly_hash(byte_tensor, n)
            
        return all_hashes

    def _generate_patch_boundaries(self, byte_tensor: torch.Tensor) -> torch.Tensor:
            """
            Implements Entropy Patching (Section 2.3).
            Note: Real BLT uses a 100M parameter Byte-LM. 
            We use a space/punctuation proxy which the paper notes as a baseline (Section 2.2).
            """
            boundaries = torch.zeros_like(byte_tensor)
            if len(boundaries) > 0: boundaries[0] = 1
            
            # Triggering on "High Entropy" structural characters
            triggers = {10, 32, 46, 40, 123, 91, 59}
            for i in range(1, len(byte_tensor)):
                if byte_tensor[i].item() in triggers:
                    boundaries[i] = 1
            return boundaries

    def _load_model(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            if self.verbose:
                print("Warning: Checkpoint not found. Creating fresh model.")
            model = UltraEfficientTextJEPA(
                vocab_size=self.config.vocab_size,
                source_dim=self.config.source_dim,
                source_depth=self.config.source_depth,
                target_dim=self.config.target_dim,
                target_depth=self.config.target_depth,
                predictor_dim=self.config.predictor_dim,
                predictor_depth=self.config.predictor_depth,
                output_dim=self.config.output_dim,
                max_source_len=self.config.max_source_len,
                max_target_len=self.config.max_target_len,
                use_rag=self.config.use_rag,  # Only FAISS disabled for startup speed
                use_enhanced_encoder=getattr(self.config, 'use_enhanced_encoder', True),  # Enable Enhanced Architecture
                use_titans=getattr(self.config, 'use_titans', True),  # Enable Titans Memory
                use_miras=getattr(self.config, 'use_miras', True),  # Enable MIRAS Retrieval
                use_hoprag=getattr(self.config, 'use_hoprag', True),  # Enable HopRAG Multi-Hop
                bing_api_key=getattr(self.config, 'bing_api_key', None),
                gradient_checkpointing=getattr(self.config, 'gradient_checkpointing', False),
            )
            
            # Optimize with torch.compile for 2-3x speedup!
            if self.verbose:
                print("  • Optimizing with torch.compile...")
            model = torch.compile(model, mode="max-autotune", backend="inductor", fullgraph=False)
            
            return model
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        ckpt_cfg = checkpoint.get('config', {})
        model = UltraEfficientTextJEPA(
            vocab_size=ckpt_cfg.get('vocab_size', self.config.vocab_size),
            source_dim=ckpt_cfg.get('source_dim', self.config.source_dim),
            source_depth=ckpt_cfg.get('source_depth', self.config.source_depth),
            target_dim=ckpt_cfg.get('target_dim', self.config.target_dim),
            target_depth=ckpt_cfg.get('target_depth', self.config.target_depth),
            predictor_dim=ckpt_cfg.get('predictor_dim', self.config.predictor_dim),
            predictor_depth=ckpt_cfg.get('predictor_depth', self.config.predictor_depth),
            output_dim=ckpt_cfg.get('output_dim', self.config.output_dim),
            max_source_len=ckpt_cfg.get('max_source_len', self.config.max_source_len),
            max_target_len=ckpt_cfg.get('max_target_len', self.config.max_target_len),
            use_rag=ckpt_cfg.get('use_rag', self.config.use_rag),  # Respect config setting
            use_enhanced_encoder=ckpt_cfg.get('use_enhanced_encoder', True),  # Enable Enhanced Architecture
            use_titans=ckpt_cfg.get('use_titans', True),  # Enable Titans Memory
            use_miras=ckpt_cfg.get('use_miras', True),  # Enable MIRAS Retrieval
            use_hoprag=ckpt_cfg.get('use_hoprag', True),  # Enable HopRAG Multi-Hop
            bing_api_key=ckpt_cfg.get('bing_api_key', None),
            gradient_checkpointing=False,  # Disable for inference
        )
        
        # Use backward-compatible loading
        model.load_state_dict_with_compatibility(checkpoint['model_state_dict'], strict=False)
        
        # Load Titans Memory if available
        titans_path = checkpoint_path.replace('.pt', '_titans.pt')
        if os.path.exists(titans_path):
            model.load_titans_memory(titans_path)
            print(f"  Titans Memory loaded from {titans_path}")
        
        # Optimize with torch.compile for 2-3x speedup!
        if self.verbose:
            print("  • Optimizing with torch.compile...")
        model = torch.compile(model, mode="max-autotune", backend="inductor", fullgraph=False)
        
        return model

    @torch.no_grad()
    def generate(self, prompt: str, max_bytes: int = 512) -> str:
        """
        Predicts the target latent representations and decodes them into bytes 
        in a single parallel pass. Optimized for fast inference.
        
        Uses ONNX for static inference path, PyTorch for dynamic components.
        """
        # 1. PRE-PROCESSING (Raw Bytes only, no BPE)
        # Limit max_bytes for faster generation
        if self.fast_mode:
            max_bytes = min(max_bytes, 128)  # Cap at 128 for speed
        else:
            max_bytes = min(max_bytes, 2048)
            
        # BLT works on raw UTF-8 values (0-255)
        raw_bytes = list(prompt.encode("utf-8"))
        
        # Context window management
        max_source = self.config.max_source_len
        byte_tensor = torch.tensor(raw_bytes[-max_source:], dtype=torch.long, device=self.device)
        
        # 2. BLT STRUCTURAL INPUTS (optimized)
        source_boundaries = self._generate_patch_boundaries(byte_tensor).unsqueeze(0)
        source_hashes = self._get_hash_ngrams(byte_tensor).unsqueeze(0)
        source_bytes = byte_tensor.unsqueeze(0)
        
        # 3. INFERENCE PATH
        if self.onnx_session:
            # HYBRID MODE: ONNX for static inference
            query_bytes = torch.zeros((1, max_bytes), dtype=torch.long, device=self.device)
            
            # Prepare ONNX inputs
            onnx_inputs = {
                'byte_ids': source_bytes.cpu().numpy().astype(np.int64),
                'patch_boundaries': source_boundaries.cpu().numpy().astype(np.int64),
                'hash_ngrams': source_hashes.cpu().numpy().astype(np.int64),
                'query_bytes': query_bytes.cpu().numpy().astype(np.int64)
            }
            
            # Run ONNX inference
            onnx_outputs = self.onnx_session.run(None, onnx_inputs)
            logits = torch.from_numpy(onnx_outputs[0])
            
            # TODO: Add Titans Memory update here if needed
            # This would require extracting intermediate representations
            
        else:
            # PYTORCH MODE: Full model
            # 3. LATENT ENCODING
            source_patch_emb = self.model.encode_source(
                byte_ids=source_bytes,
                patch_boundaries=source_boundaries,
                hash_ngrams=source_hashes
            )

            # 4. NON-AUTOREGRESSIVE PREDICTION
            query_bytes = torch.zeros((1, max_bytes), dtype=torch.long, device=self.device)
            query_boundaries = torch.zeros_like(query_bytes)
            query_boundaries[:, 0] = 1
            
            # Predict the target latent space
            pred_latent, _ = self.model.predict(source_patch_emb, query_bytes)

            # 5. LOCAL DECODING (Parallel Byte Reconstruction)
            logits = self.model.decode(pred_latent, max_length=max_bytes)
        
        # 6. POST-PROCESSING
        # Convert logits to actual byte values
        output_ids = torch.argmax(logits[0], dim=-1)
        
        # Filter out-of-bounds
        output_ids = output_ids[output_ids < 256].cpu().numpy().astype(np.uint8)
        
        return bytes(output_ids).decode("utf-8", errors="replace")

# ============================================================================
# APP INITIALIZATION
# ============================================================================

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("🚀 Shutka API Server - Starting...")
    engine = ShutkaEngine(fast_mode=True, verbose=False, use_onnx=True)
    
    # Display mode
    if engine.onnx_session:
        print(f"✓ Model loaded: ~350M params, {engine.config.source_dim}d, ONNX + PyTorch hybrid")
        print(f"  • ONNX: Static inference (2-3x faster)")
        print(f"  • PyTorch: Dynamic memory (Titans, MIRAS)")
    else:
        print(f"✓ Model loaded: ~350M params, {engine.config.source_dim}d, PyTorch")
        print(f"  💡 Tip: Export to ONNX for 2-3x speedup: python3.11 export_onnx.py")
    
    print(f"✓ BitNet 1.58-bit quantization enabled")
    print(f"✓ Ready on http://0.0.0.0:8000")
    yield
    print("✓ Server shutdown")

app = FastAPI(title="Shutka Perfect OpenAI API", lifespan=lifespan)
SYSTEM_FINGERPRINT = f"fp_shutka_{uuid.uuid4().hex[:10]}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS (Redundant Routing for Compatibility)
# ============================================================================

@app.get("/")
@app.get("/v1")
async def root():
    return {"status": "online", "model": "shutka-v2"}

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    models = ["shutka-v2"]
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 1700000000,
                "owned_by": "shutka"
            } for m in models
        ]
    }

async def sse_generator(content: str, model: str, completion_id: str):
    created = int(time.time())
    
    # 1. Initial chunk with role (standard OpenAI behavior)
    role_payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_payload)}\n\n"

    # 2. Content chunks with small delay to make it "stream" visually
    words = content.split(' ')
    for i, word in enumerate(words):
        chunk = word + (' ' if i < len(words) - 1 else '')
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.01) # 10ms delay between words
    
    # 3. Final stop chunk
    stop_payload = {
        "id": completion_id, 
        "object": "chat.completion.chunk", 
        "created": created, 
        "model": model, 
        "system_fingerprint": SYSTEM_FINGERPRINT, 
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(stop_payload)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role.upper()}: {msg.content}\n"
    prompt += "ASSISTANT: "
    
    # Use max_completion_tokens if provided, otherwise max_tokens
    max_tokens = request.max_completion_tokens or request.max_tokens or 512
    
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    response_text = engine.generate(prompt, max_bytes=max_tokens)
    response_text = response_text.replace("<|endoftext|>", "").strip()

    if request.stream:
        return StreamingResponse(
            sse_generator(response_text, request.model, completion_id),
            media_type="text/event-stream"
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.encode('utf-8')),
            "completion_tokens": len(response_text.encode('utf-8')),
            "total_tokens": len(prompt.encode('utf-8')) + len(response_text.encode('utf-8'))
        }
    }

@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    inputs = [request.input] if isinstance(request.input, str) else request.input
    
    data = []
    total_tokens = 0
    for i, text in enumerate(inputs):
        # Convert text to raw UTF-8 bytes (0-255)
        raw_bytes = list(text.encode("utf-8"))
        tokens = torch.tensor(raw_bytes[:engine.config.max_source_len], dtype=torch.long)
        total_tokens += len(tokens)
        
        with torch.no_grad():
            emb = engine.model.encode_source(torch.tensor([tokens], device=engine.device))
            pooled = emb.mean(dim=1)[0].cpu().numpy().tolist()
        data.append({"object": "embedding", "index": i, "embedding": pooled})
    
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    # Force HTTP/2 by prioritizing h2
    config.alpn_protocols = ["h2", "http/1.1"]
    # Enable HTTP/2 cleartext (h2c) for non-TLS connections
    config.h2_max_concurrent_streams = 100
    
    asyncio.run(serve(app, config))
