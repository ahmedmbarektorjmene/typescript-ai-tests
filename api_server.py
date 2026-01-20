import os
import torch
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

# ONNX Runtime for optimized inference
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import Shutka components
from models.shutka import UltraEfficientTextJEPA
from config import TrainingConfig


N_MIN = 3
N_MAX = 8
HASH_TABLE_SIZE = 370000


def rolling_poly_hash(
    bytes_tensor: torch.Tensor, n: int, prime: int = 1000003
) -> torch.Tensor:
    length = bytes_tensor.size(0)
    hashes = torch.zeros(length, dtype=torch.long)
    if length < n:
        return hashes

    current_hash = 0
    for i in range(n):
        current_hash = (current_hash * 256 + bytes_tensor[i].item()) % HASH_TABLE_SIZE

    hashes[n - 1] = current_hash
    power = pow(256, n - 1, HASH_TABLE_SIZE)
    for i in range(n, length):
        current_hash = (
            current_hash - bytes_tensor[i - n].item() * power
        ) % HASH_TABLE_SIZE
        current_hash = (current_hash * 256 + bytes_tensor[i].item()) % HASH_TABLE_SIZE
        hashes[i] = current_hash

    return hashes


# ============================================================================
# API MODELS
# ============================================================================


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "shutka-v2"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False


class EmbeddingRequest(BaseModel):
    model: str = "shutka-v2"
    input: Union[str, List[str]]


# ============================================================================
# MODEL ENGINE
# ============================================================================


class ShutkaEngine:
    def __init__(
        self,
        checkpoint_path: str = "models/shutka.pt",
        onnx_path: str = "models/shutka.onnx",
        fast_mode: bool = True,
        verbose: bool = False,
        use_onnx: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.config = TrainingConfig()

        if fast_mode:
            self.config.source_dim = 512
            self.config.source_depth = 24

        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device).eval()

        if hasattr(self.model, "precompute_mhc"):
            self.model.precompute_mhc()

        self.vocab_size = self.model.vocab_size

    def _get_hash_ngrams(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        length = byte_tensor.size(0)
        all_hashes = torch.zeros((length, N_MAX - N_MIN + 1), dtype=torch.long)
        for idx, n in enumerate(range(N_MIN, N_MAX + 1)):
            all_hashes[:, idx] = rolling_poly_hash(byte_tensor, n)
        return all_hashes

    def _load_model(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            return UltraEfficientTextJEPA(
                vocab_size=self.config.vocab_size,
                source_dim=self.config.source_dim,
                source_depth=self.config.source_depth,
            )

        # Determine if it's a TorchScript model or state dict
        try:
            model = torch.jit.load(checkpoint_path, map_location=self.device)
            return model
        except:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            ckpt_cfg = checkpoint.get("config", {})
            model = UltraEfficientTextJEPA(
                vocab_size=ckpt_cfg.get("vocab_size", self.config.vocab_size),
                source_dim=ckpt_cfg.get("source_dim", self.config.source_dim),
                source_depth=ckpt_cfg.get("source_depth", self.config.source_depth),
                engram_vocab_size=ckpt_cfg.get("engram_vocab_size", 370000),
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            return model

    def _incremental_poly_hash(
        self, current_hash: int, out_byte: int, in_byte: int, n: int
    ) -> int:
        power = pow(256, n - 1, HASH_TABLE_SIZE)
        new_hash = (current_hash - out_byte * power) % HASH_TABLE_SIZE
        new_hash = (new_hash * 256 + in_byte) % HASH_TABLE_SIZE
        return new_hash

    @torch.no_grad()
    def generate(self, prompt: str, max_bytes: int = 512) -> str:
        raw_bytes = list(prompt.encode("utf-8"))
        input_ids = torch.tensor(
            raw_bytes, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        input_hashes = self._get_hash_ngrams(input_ids[0]).unsqueeze(0).to(self.device)

        generated_bytes = []
        past_key_values = None

        # History for rolling hash
        rolling_window = raw_bytes[:]
        if len(rolling_window) < N_MAX:
            rolling_window = [0] * (N_MAX - len(rolling_window)) + rolling_window

        current_hashes = input_hashes[0, -1].tolist()

        for i in range(max_bytes):
            logits, past_key_values = self.model(
                input_ids, hash_ngrams=input_hashes, past_key_values=past_key_values
            )
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            if next_token_id >= 256:
                break
            generated_bytes.append(next_token_id)

            # Incremental updates
            input_ids = torch.tensor([[next_token_id]], device=self.device)
            new_hashes = []
            for idx, n in enumerate(range(N_MIN, N_MAX + 1)):
                out_byte = rolling_window[-n]
                new_h = self._incremental_poly_hash(
                    current_hashes[idx], out_byte, next_token_id, n
                )
                new_hashes.append(new_h)

            input_hashes = torch.tensor([new_hashes], device=self.device).unsqueeze(0)
            current_hashes = new_hashes
            rolling_window.append(next_token_id)
            if len(rolling_window) > N_MAX:
                rolling_window = rolling_window[-N_MAX:]

        return bytes(generated_bytes).decode("utf-8", errors="replace")


# ============================================================================
# APP
# ============================================================================

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("🚀 Shutka API Server - Starting...")
    engine = ShutkaEngine(verbose=True)
    print("✓ Ready on http://0.0.0.0:8000")
    yield


app = FastAPI(title="Shutka Perfect OpenAI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = (
        "".join([f"{msg.role.upper()}: {msg.content}\n" for msg in request.messages])
        + "ASSISTANT: "
    )
    max_tokens = request.max_completion_tokens or request.max_tokens or 512
    response_text = engine.generate(prompt, max_bytes=max_tokens).strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    config.alpn_protocols = ["h2", "http/1.1"]
    asyncio.run(serve(app, config))
