import time
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from pathlib import Path
from torch.utils.data import DataLoader
from model.config import ARESConfig
from training.checkpoint import CheckpointManager

class ARESTrainer:
    """
    Research-grade training orchestration engine for ARES-Base.
    Handles gradient accumulation, clipping, validation loops, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        config: ARESConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "experiments/checkpoints",
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Training hyper-parameters & intervals
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = max(1, gradient_accumulation_steps)
        
        # Internal tracking state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        
        # Create checkpoint directory if it does not exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        print(f"[ARESTrainer] Initializing training on device: {self.device.upper()}")
        print(f"[ARESTrainer] Total Epochs: {num_epochs} | Grad Accumulation Steps: {self.grad_accum_steps}")
        
        history = {"train_loss": [], "val_loss": []}
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                # 1. Move batch to target device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # 2. Forward Pass
                _, loss, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False  # Disabled during training to save VRAM
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
                
                # 3. Backward Pass
                loss.backward()
                
                epoch_loss += loss.item() * self.grad_accum_steps
                
                # 4. Optimizer Step (executed every N accumulation steps)
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_dataloader):
                    # Clip gradients to prevent exploding gradient instability in GPT-2
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    if self.optimizer:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    if self.scheduler:
                        self.scheduler.step()
                        
                    self.global_step += 1
                    
                    # ---------------------------------------------------------
                    # Logging
                    # ---------------------------------------------------------
                    if self.global_step % self.log_interval == 0:
                        elapsed = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0] if self.scheduler else (
                            self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
                        )
                        print(
                            f"Epoch [{epoch+1}/{num_epochs}] | Step [{self.global_step}] | "
                            f"Loss: {loss.item() * self.grad_accum_steps:.4f} | "
                            f"LR: {lr:.2e} | Time: {elapsed:.2f}s"
                        )
                    
                    # ---------------------------------------------------------
                    # Evaluation
                    # ---------------------------------------------------------
                    if self.val_dataloader and self.global_step % self.eval_interval == 0:
                        val_loss = self.evaluate()
                        history["val_loss"].append(val_loss)
                        
                        try:
                            perplexity = math.exp(val_loss)
                            perplexity_str = f"{perplexity:.2f}"
                        except OverflowError:
                            perplexity_str = "inf"

                        print(
                            f"--> [Validation] Step {self.global_step} | "
                            f"Val Loss: {val_loss:.4f} | Perplexity: {perplexity_str}"
                        )
                        
                        # Save best checkpoint
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save("best_model.pt")
                            
                        self.model.train()  # Return to training mode after evaluation

                    # ---------------------------------------------------------
                    # Checkpointing
                    # ---------------------------------------------------------
                    if self.global_step % self.save_interval == 0:
                        self._save(f"checkpoint_step_{self.global_step}.pt")
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            history["train_loss"].append(avg_epoch_loss)
            print(f"=== Epoch {epoch+1} Completed | Average Train Loss: {avg_epoch_loss:.4f} ===")
            
            # Save epoch checkpoint
            self._save(f"checkpoint_epoch_{epoch+1}.pt")
            
        print("[ARESTrainer] Training loop completed successfully.")
        return history

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Runs evaluation over the validation dataset without calculating gradients.
        
        Returns:
            float: Mean validation CrossEntropy loss.
        """
        if not self.val_dataloader:
            return 0.0
            
        self.model.eval()
        total_val_loss = 0.0
        total_steps = 0
        
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            _, loss, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )
            
            total_val_loss += loss.item()
            total_steps += 1
            
        return total_val_loss / max(1, total_steps)

    def _save(self, filename: str) -> None:
        """Internal helper to dispatch saving logic to CheckpointManager."""
        if not self.optimizer:
            return
            
        filepath = self.output_dir / filename
        CheckpointManager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            filepath=filepath,
            config=self.config
        )