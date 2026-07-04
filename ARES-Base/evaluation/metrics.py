import torch
import torch.nn as nn
from typing import Dict, Optional

class EvaluationMetrics:
    #standard token level eval metrics
    @staticmethod
    def compute_token_accuracy(
        logits:torch.Tensor,
        labels:torch.Tensor,
        ignore_index: int=-100
    )->float:
        #returns percentage of correctly predicted tokens
        shift_logits=logits[...,:-1,:].contiguous()
        shift_labels=labels[...,:1:].contiguous()

        predictions=torch.argmax(shift_logits,dim=-1)

        mask=shift_labels!=ignore_index

        #compute acc only on valid semantic tokens
        correct_tokens=(predictions==shift_labels)& mask
        total_valid_tokens=mask.sum().item()

        if total_valid_tokens==0:
            return 0.0
        
        return (correct_tokens.sum().item() / total_valid_tokens) * 100.0
    
    @staticmethod
    def compute_cross_entropy_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    )->float:
        #compute mean cross entropy loss over valid seq tokens
        shift_logits=logits[...,:-1,:].contiguous()
        shift_labels=labels[...,:1:].contiguous()

        loss_fct=nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
        loss=loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss.item()