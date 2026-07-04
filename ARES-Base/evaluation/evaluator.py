"""
This class acts as the centralized benchmarking coordinator, allowing us to generate clean, structured evaluation summaries during experiment runs.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tokenizer.tokenizer import BaseTokenizer
from evaluation.metrics import EvaluationMetrics
from evaluation.perplexity import PerplexityCalculator

class ARESEvaluator:
    #orchestration engine for ARES-Base model eval and benchmark reporting
    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.ppl_calculator = PerplexityCalculator(model, tokenizer, device=device)

    @torch.no_grad()
    def evaluate_dataloader(self, val_dataloader: DataLoader) -> Dict[str, float]:
        #returns Loss, Perplexity, and Token Accuracy.

        print(f"[ARESEvaluator] Running full benchmark evaluation on {self.device.upper()}...")

        total_loss = 0.0
        total_accuracy = 0.0
        total_steps = 0

        for batch in val_dataloader:
            input_ids=batch['input_ids'].to(self.device)
            labels=batch['labels'].to(self.device)

            logits,loss=self.model(input_ids=input_ids, labels=labels, use_cache=False)

            acc = EvaluationMetrics.compute_token_accuracy(
                logits=logits, 
                labels=labels, 
                ignore_index=self.tokenizer.pad_token_id
            )

            total_loss += loss.item()
            total_accuracy += acc
            total_steps += 1

        if total_steps == 0:
            raise ValueError("Validation DataLoader is empty.")
            
        mean_loss = total_loss / total_steps
        mean_acc = total_accuracy / total_steps
        perplexity = math.exp(mean_loss)

        results = {
            "val_loss": round(mean_loss, 4),
            "perplexity": round(perplexity, 2),
            "token_accuracy_pct": round(mean_acc, 2)
        }

        print("================= EVALUATION RESULTS =================")
        print(f" Mean Loss        : {results['val_loss']}")
        print(f" Perplexity (PPL) : {results['perplexity']}")
        print(f" Token Accuracy   : {results['token_accuracy_pct']}%")
        print("======================================================")

        return results
        
    def evaluate_document_perplexity(self, text:str,stride:int=512)->float:
        self.ppl_calculator.stride = stride
        return self.ppl_calculator.compute_from_text(text)