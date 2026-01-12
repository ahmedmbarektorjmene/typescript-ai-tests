import os
import torch
import torch.nn.functional as F
import tiktoken
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Generator, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

# Import Shutka components
from models.shutka import UltraEfficientTextJEPA
from config import TrainingConfig

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
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    model: str = "shutka-v2"
    input: Union[str, List[str]]

class MemoryRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# MODEL ENGINE
# ============================================================================

class ShutkaEngine:
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Shutka model onto {self.device}...")
        
        self.config = TrainingConfig()
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device).eval()
        
        # STRICT: Match model vocab size
        self.vocab_size = self.model.x_encoder.token_embed.num_embeddings
        
        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
            print(f"  Tokenizer: cl100k_base (Efficient Vocab: {self.vocab_size})")
        except:
            self.enc = None
            print("  Error: Could not load tiktoken cl100k_base")

    def _load_model(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            print("Warning: Checkpoint not found. Creating fresh model.")
            return UltraEfficientTextJEPA(
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
                use_rag=self.config.use_rag
            )
        
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
            use_rag=ckpt_cfg.get('use_rag', self.config.use_rag)
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.enc: return "Error: Tokenizer not loaded."
        
        # continue sometimes sends huge max_tokens. Clamp it.
        max_tokens = min(max_tokens, 2048) 
        
        tokens = self.enc.encode(prompt)
        # Limit source to context window.
        # Total (source + target) should be roughly under 8192
        max_source = 4096 
        tokens = tokens[-max_source:]
        
        # SAFETY: Clamp to vocab
        tokens = [min(t, self.vocab_size - 1) for t in tokens]
        source_tokens = torch.tensor([tokens], device=self.device)
        
        source_emb = self.model.encode_source(source_tokens)
        query_tokens = torch.zeros((1, max_tokens), dtype=torch.long, device=self.device)
        query_emb = self.model.y_encoder.token_embed(query_tokens)
        
        pred_emb, _ = self.model.predict(source_emb, query_emb)
        logits = self.model.decode(pred_emb, max_length=max_tokens)
        
        output_ids = torch.argmax(logits[0], dim=-1)
        output_ids = output_ids[output_ids < self.vocab_size].cpu().numpy()
        
        return self.enc.decode(output_ids.tolist())

# ============================================================================
# APP INITIALIZATION
# ============================================================================

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ShutkaEngine()
    yield

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
    
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    response_text = engine.generate(prompt, max_tokens=request.max_tokens)
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
            "prompt_tokens": len(engine.enc.encode(prompt)),
            "completion_tokens": len(engine.enc.encode(response_text)),
            "total_tokens": 0
        }
    }

@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    inputs = [request.input] if isinstance(request.input, str) else request.input
    
    data = []
    for i, text in enumerate(inputs):
        tokens = engine.enc.encode(text)
        tokens = [min(t, engine.vocab_size - 1) for t in tokens]
        with torch.no_grad():
            emb = engine.model.encode_source(torch.tensor([tokens], device=engine.device))
            pooled = emb.mean(dim=1)[0].cpu().numpy().tolist()
        data.append({"object": "embedding", "index": i, "embedding": pooled})
        
    return {"object": "list", "data": data, "model": request.model}

@app.post("/v1/memory")
@app.post("/memory")
async def add_memory(request: MemoryRequest):
    """Dynamically add new context to the model's RAG bank"""
    print(f"LOG: API Request -> Add Memory (text_len={len(request.text)})")
    
    if not engine.model.predictor.use_rag:
        raise HTTPException(status_code=400, detail="RAG is disabled on this model instance.")

    # 1. Generate embedding for the new text
    tokens = engine.enc.encode(request.text)
    tokens = [min(t, engine.vocab_size - 1) for t in tokens]
    
    with torch.no_grad():
        # Encode with source encoder (used for queries)
        emb = engine.model.encode_source(torch.tensor([tokens], device=engine.device))
        # Project to predictor dimension (where RAG lives)
        proj_emb = engine.model.predictor.source_proj(emb)
        # Use mean as representation
        vec = proj_emb.mean(dim=1)

    # 2. Add to FAISS bank
    # Ensure memory bank is accessible
    bank = engine.model.predictor.memory_bank
    ids = bank.add_memory(vec, [request.text])
    
    return {
        "status": "success",
        "ids": ids,
        "message": f"Successfully added {len(ids)} memory entry."
    }

if __name__ == "__main__":
    # Hypercorn configuration to enable HTTP/2 (including h2c)
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    config.alpn_protocols = ["h2", "http/1.1"]
    
    print("Starting server with Hypercorn (HTTP/2 enabled)...")
    asyncio.run(serve(app, config))
