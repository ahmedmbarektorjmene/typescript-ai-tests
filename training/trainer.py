"""
Trainer class for model training with GPU/CPU optimization

Supports:
- Modern GPU: torch.compile, BF16/FP16 mixed precision, TF32 tensor cores
- CPU: Quantization-aware training
- Auto-detection of best settings
"""

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from tqdm import tqdm
from typing import Optional

from models.shutka import Optimizer

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# Check for modern GPU features
CUDA_AVAILABLE = torch.cuda.is_available()
BF16_SUPPORTED = CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")


class Trainer:
    """
    Trainer for sequence models with GPU/CPU optimization

    Modern GPU features:
    - torch.compile for 10-30% speedup (PyTorch 2.0+)
    - BF16/FP16 mixed precision for faster training
    - TF32 tensor cores on Ampere+ GPUs
    - Automatic gradient scaling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: Optional[torch.device] = None,
        use_compile: bool = True,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,  # T4 optimization
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.use_mixed_precision = use_mixed_precision and CUDA_AVAILABLE
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps  # T4 optimization
        )

        # Validate configuration
        if hasattr(config, "validate"):
            is_valid, errors = config.validate()
            if not is_valid:
                print("\n⚠️  Configuration validation warnings:")
                for error in errors:
                    if not error.startswith("WARNING"):
                        print(f"  ERROR: {error}")
                        raise ValueError(f"Invalid configuration: {error}")
                    else:
                        print(f"  {error}")

        # Device: auto-detect GPU if available
        if device is None:
            if CUDA_AVAILABLE:
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.device = torch.device("cpu")
                print("❌ ERROR: GPU not available!")
                print("This model requires a GPU for training (T4, A100, etc.)")
                print("Please run on Kaggle, Colab, or a machine with CUDA GPU.")
                raise RuntimeError(
                    "GPU required for training. CPU training is not supported."
                )
        else:
            self.device = device
            if not CUDA_AVAILABLE and device.type == "cuda":
                raise RuntimeError("CUDA device requested but CUDA is not available.")

        # Enable TF32 for tensor cores (RTX 30xx/40xx, A100, T4)
        if CUDA_AVAILABLE:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = (
                True  # Auto-tune kernels for better performance
            )
            torch.backends.cudnn.deterministic = (
                False  # Allow non-deterministic for speed
            )
            print("  TF32 tensor cores: ENABLED")
            print("  cuDNN auto-tuning: ENABLED")

            # Set memory allocator settings for better GPU memory management
            if hasattr(torch.cuda, "memory"):
                torch.cuda.empty_cache()  # Clear cache before training
                # Enable memory pool for faster allocations
                torch.cuda.set_per_process_memory_fraction(
                    0.95
                )  # Use up to 95% of GPU memory

        self.model.to(self.device)

        # Apply torch.compile for speedup (PyTorch 2.0+)
        if use_compile and TORCH_COMPILE_AVAILABLE and CUDA_AVAILABLE:
            try:
                print("Compiling model with torch.compile...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  torch.compile: ENABLED")
            except Exception as e:
                print(f"  torch.compile failed: {e}")

        # Setup optimizer
        # Always use custom Optimizer as requested
        galore_params = []
        standard_params = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if p.dim() >= 2:
                    galore_params.append(p)
                else:
                    standard_params.append(p)

        self.optimizer = Optimizer(
            galore_params + standard_params,
            lr=config.learning_rate,
            rank=getattr(config, "rank", 128),
            update_freq=getattr(config, "update_freq", 200),
            cans_steps=getattr(config, "cans_steps", 5),
        )
        print(f"  Optimizer: Custom (rank={getattr(config, 'rank', 128)})")

        # Setup mixed precision
        if self.use_mixed_precision:
            # Modern GradScaler API
            # Start with lower scale to prevent "Step 0 NaN" on T4 FP16
            self.scaler = torch.amp.GradScaler("cuda", init_scale=2048.0)

            # Optimized Dtype: Tesla T4 (Turing) supports FP16 much better than BF16.
            is_ampere_or_newer = (
                CUDA_AVAILABLE and torch.cuda.get_device_capability(0)[0] >= 8
            )
            if BF16_SUPPORTED and is_ampere_or_newer:
                self.autocast_dtype = torch.bfloat16
            else:
                self.autocast_dtype = torch.float16

            print(f"  Mixed precision: {self.autocast_dtype}")
        else:
            self.scaler = None
            self.autocast_dtype = None
            print("  Mixed precision: DISABLED")

        # Learning rate scheduler
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Memory and throughput tracking
        self.start_time = None
        self.total_tokens = 0
        self.memory_stats = {
            "peak_memory_mb": 0,
            "current_memory_mb": 0,
            "tokens_per_sec": 0,
        }

        # Create checkpoint directory
        # Sanitize input: if user passed a file path (e.g., best_model.pt), extract dir
        if config.checkpoint_dir.endswith(".pt"):
            print(
                f"Warning: checkpoint_dir '{config.checkpoint_dir}' looks like a file. Using parent directory."
            )
            config.checkpoint_dir = os.path.dirname(config.checkpoint_dir)
            if not config.checkpoint_dir:
                config.checkpoint_dir = "."

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Initialize memory building counter
        self.memory_build_steps = 0

        # Print optimization summary
        self._print_optimization_summary()

    def _log_memory_and_throughput(self, num_tokens: int):
        """Log memory usage and throughput metrics"""
        if CUDA_AVAILABLE:
            # GPU memory tracking
            current_memory = torch.cuda.memory_allocated() / 1e6  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

            self.memory_stats["current_memory_mb"] = current_memory
            self.memory_stats["peak_memory_mb"] = max(
                peak_memory, self.memory_stats["peak_memory_mb"]
            )
        elif PSUTIL_AVAILABLE:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1e6  # MB
            self.memory_stats["current_memory_mb"] = current_memory
            self.memory_stats["peak_memory_mb"] = max(
                current_memory, self.memory_stats["peak_memory_mb"]
            )

        # Throughput tracking
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.total_tokens += num_tokens
            self.memory_stats["tokens_per_sec"] = (
                self.total_tokens / elapsed_time if elapsed_time > 0 else 0
            )

    def _print_optimization_summary(self):
        """Print summary of enabled optimizations"""
        print("\n" + "=" * 50)
        print("OPTIMIZATION SUMMARY")
        print("=" * 50)
        print(f"Device: {self.device}")
        if CUDA_AVAILABLE:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(
            f"Mixed Precision: {self.autocast_dtype if self.use_mixed_precision else 'Disabled'}"
        )
        print(
            f"torch.compile: {'Enabled' if TORCH_COMPILE_AVAILABLE and CUDA_AVAILABLE else 'Disabled'}"
        )
        print(f"TF32: {'Enabled' if CUDA_AVAILABLE else 'N/A'}")
        print("=" * 50 + "\n")

    def _check_nan_gradients(self) -> bool:
        """Check for NaN or Inf gradients in model parameters"""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False

    def build_memory_from_batch(self, source_tokens, target_tokens, loss):
        """Build FAISS memory from training batch"""
        # Check if memory bank is available
        if not hasattr(self.model, "predictor"):
            return

        if (
            not hasattr(self.model.predictor, "memory_bank")
            or self.model.predictor.memory_bank is None
        ):
            return

        # Only build memory occasionally to avoid overhead
        # For small datasets, we want to build frequently
        if self.memory_build_steps % 1 != 0:
            return

        # Encode source and target to get embeddings
        with torch.no_grad():
            source_emb = self.model.encode_source(source_tokens)

            # Mean pool source embeddings for memory
            source_pooled = source_emb.mean(dim=1)  # [batch, hidden_dim]

            # Create memory texts from token sequences (simplified)
            memory_texts = []
            for i in range(source_tokens.shape[0]):
                # Convert first few tokens to a simple text representation
                tokens = (
                    source_tokens[i][:64].cpu().numpy()
                )  # First 64 tokens as context
                text = str(tokens.tolist())  # Store as stringified list for recovery
                memory_texts.append(text)

            # Add to memory bank
            self.model.predictor.memory_bank.add_memory(source_pooled, memory_texts)

        self.memory_build_steps += 1

    def train_epoch(self, epoch: int):
        """Train for one epoch with mixed precision support"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Start timing for throughput
        if self.start_time is None:
            self.start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device - VL-JEPA format (already collated and padded)
            source_tokens = batch["source_patches"].to(self.device, non_blocking=True)
            target_tokens = batch["target_patches"].to(self.device, non_blocking=True)

            # Create query tokens (same as target for simplicity in JEPA)
            query_tokens = target_tokens

            # Forward pass with mixed precision
            self.optimizer.zero_grad(
                set_to_none=True
            )  # More efficient than zero_grad()

            if self.use_mixed_precision and self.scaler is not None:
                # Mixed precision forward pass
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    loss, _, _ = self.model(
                        source_tokens=source_tokens,
                        query_tokens=query_tokens,
                        target_tokens=target_tokens,
                    )

                # Build FAISS memory (outside autocast)
                self.build_memory_from_batch(source_tokens, target_tokens, loss)

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"\nWarning: NaN/Inf loss detected at step {self.global_step}. Skipping batch."
                    )
                    continue

                # Mixed precision backward pass
                self.scaler.scale(loss).backward()

                # NUCLEAR SYNC: Fix dtype mismatches between parameters and gradients.
                # bitsandbytes mutates weights to float16 during forward, but GradScaler
                # produces float32 gradients. This causes a crash in the optimizer (assignment error).
                # We synchronize everything back to float32 before unscaling/stepping.
                for p in self.model.parameters():
                    if p.grad is not None:
                        if p.dtype == torch.float16 or p.grad.dtype == torch.float16:
                            # Re-sync to float32 for optimizer compatibility
                            p.data = p.data.to(torch.float32)
                            p.grad.data = p.grad.data.to(torch.float32)

                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = self._check_nan_gradients()
                if has_nan_grad:
                    print(
                        f"\nWarning: NaN/Inf gradients at step {self.global_step}. Skipping update."
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.scaler.step(self.optimizer)

                self.scaler.update()
            else:
                # Standard precision forward pass (CPU or explicit disable)
                loss, _, _ = self.model(
                    source_tokens=source_tokens,
                    query_tokens=query_tokens,
                    target_tokens=target_tokens,
                )

                # Build FAISS memory
                self.build_memory_from_batch(source_tokens, target_tokens, loss)

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"\nWarning: NaN/Inf loss detected at step {self.global_step}. Skipping batch."
                    )
                    continue

                # Standard backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = self._check_nan_gradients()
                if has_nan_grad:
                    print(
                        f"\nWarning: NaN/Inf gradients at step {self.global_step}. Skipping update."
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.step()
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log memory and throughput
            num_tokens = source_tokens.numel()
            self._log_memory_and_throughput(num_tokens)

            # Update progress bar with memory and throughput stats
            loss_val = (
                loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0
            )
            avg_loss_val = total_loss / num_batches if num_batches > 0 else 0.0
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "avg": f"{avg_loss_val:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                    "mem": f"{self.memory_stats['current_memory_mb']:.0f}MB",
                    "tok/s": f"{self.memory_stats['tokens_per_sec']:.0f}",
                }
            )

            # Save checkpoint periodically
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Evaluate periodically
            if self.global_step % self.config.eval_every == 0:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            # VL-JEPA format (already collated and padded)
            source_tokens = batch["source_patches"].to(
                self.device
            )  # (batch_size, seq_len)
            target_tokens = batch["target_patches"].to(
                self.device
            )  # (batch_size, seq_len)
            query_tokens = target_tokens

            # Forward pass - VL-JEPA training
            loss, _, _ = self.model(
                source_tokens=source_tokens,
                query_tokens=query_tokens,
                target_tokens=target_tokens,
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Peak Memory: {self.memory_stats['peak_memory_mb']:.0f}MB")
            print(f"  Throughput: {self.memory_stats['tokens_per_sec']:.0f} tokens/sec")

            # Save last checkpoint (overwrites previous to save disk space)
            self.save_checkpoint("checkpoint_last.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(f"  New best model saved! (val_loss: {val_loss:.4f})")

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": (
                self.config.__dict__
                if hasattr(self.config, "__dict__")
                else str(self.config)
            ),
        }
        torch.save(checkpoint, checkpoint_path)

        # Save Titans Memory if using enhanced architecture
        if hasattr(self.model, "save_titans_memory"):
            titans_path = checkpoint_path.replace(".pt", "_titans.pt")
            self.model.save_titans_memory(titans_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint with backward compatibility"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Use backward-compatible loading
        self.model.load_state_dict_with_compatibility(
            checkpoint["model_state_dict"], strict=False
        )

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Loaded checkpoint from {checkpoint_path}")

        # Load Titans Memory if available
        titans_path = checkpoint_path.replace(".pt", "_titans.pt")
        if os.path.exists(titans_path) and hasattr(self.model, "load_titans_memory"):
            self.model.load_titans_memory(titans_path)
