#to evalute docs longer thatn our models context window (1024 tokens) without boundary distortoin, we use a sliding window with a defined stride
import math
import torch
import torch.nn as nn
from typing import List, Union
from tqdm import tqdm
from tokenizer.tokenizer import BaseTokenizer

class PerplexityCalculator:
    def __init__(
            self,
            model:nn.Module,
            tokenizer:BaseTokenizer,
            max_length:int=1024,
            stride:int=512,
            device:str="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.device = device

    @torch.no_grad()
    def compute_from_text(self, text)->float:
        """
        Evaluates the perplexity of a raw text string using sliding window decoding.
        """

        #encode text into ids
        tokenized_inputs=self.tokenizer.encode(text, add_special_tokens=False)
        input_ids=torch.tensor([tokenized_inputs], dtype=torch.long, device=self.device)

        seq_len=input_ids.size(1)
        if seq_len==0:
            return float("inf")
            
        nlls: List[torch.Tensor]=[]
        prev_end_loc=0

        #slide accross token seq
        for begin_loc in range(0,seq_len, self.stride):
            end_loc=min(begin_loc+self.max_length, seq_len)
            trg_len=end_loc-prev_end_loc #target len to eval in this window

            input_chunk=input_ids[:, begin_loc:end_loc]
            target_chunk=input_chunk.clone()

            #masking out the tokens we alr eval in prev window
            target_chunk[:,:-trg_len]=-100

            #forward pass
            _,loss,_=self.model(input_ids=input_chunk, labels=target_chunk, use_cache=False)

            #weight loss by the number of active target tokens eval in this step
            neg_log_likelihood=loss*trg_len
            nlls.append(neg_log_likelihood)

            prev_end_loc=end_loc
            if end_loc==seq_len:
                break
            
        total_nll=torch.stack(nlls).sum()
        mean_nll=total_nll/seq_len

        #PPL = exp(mean_nll)
        return math.exp(mean_nll.item())
        
    @torch.no_grad()
    def compute_from_dataloader(self, dataloader:torch.utils.data.DataLoader)->float:
        """
        Computes standard aggregate perplexity across an entire pre-packed evaluation dataset.
        """
        total_loss=0.0
        total_steps=0

        for batch in tqdm(dataloader, desc="[Perplexity] Evaluating dataset"):
            input_ids=batch['input_ids'].to(self.device)
            labels=batch["labels"].to(self.device)
                
            _,loss,_=self.model(input_ids=input_ids, labels=labels, use_cache=False)
            total_loss+=loss.item()
            total_steps+=1

        if total_steps==0:
            return float("inf")
            
        mean_loss=total_loss/total_steps
        return math.exp(mean_loss)
