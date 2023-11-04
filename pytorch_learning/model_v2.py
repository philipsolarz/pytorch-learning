import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple

def scaled_dot_product_attention(
    query: Tensor, 
    key: Tensor, 
    value: Tensor, 
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[Tensor, Tensor]:
    d_k: int = query.size(-1)
    scores: torch.Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights: Tensor = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        weights = dropout(weights)
        
    output: Tensor = weights @ value
    
    return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = query.size(0)

        # Linear projection split into h heads
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all h heads.

        # Apply scaled dot product attention
        attention_output, _ = scaled_dot_product_attention(query, key, value, mask, self.dropout)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attention_output)

        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 10000.0 Hyperparameter (increase for larger sentences, decrease for smaller sentences)

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.encoding[:x.size(0), :].detach()
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention
        attn_output = self.self_attn(src, src, src, src_mask)[0]
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward network
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            src = layer(src, src_mask=mask)

        return self.norm(src)
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int= 2048, dropout: float = 0.1):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, num_heads, dropout)
        self.attn_head_over_encoder = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        # Masked multi-head attention
        tgt2 = self.masked_attn_head(tgt, tgt, tgt, tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Multi-head attention over encoder output
        tgt2 = self.attn_head_over_encoder(tgt, memory, memory, memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward network
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        return self.norm(tgt)
    
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, d_model: int, num_heads: int, 
                 dim_feedforward: int, dropout: float, input_vocab_size: int, target_vocab_size: int):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dim_feedforward, dropout)
        self.src_tok_emb = nn.Embedding(input_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.final_linear = nn.Linear(d_model, target_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor], tgt_mask: Optional[Tensor], 
                memory_mask: Optional[Tensor], src_key_padding_mask: Optional[Tensor], 
                tgt_key_padding_mask: Optional[Tensor], memory_key_padding_mask: Optional[Tensor]) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        outs = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return F.softmax(self.final_linear(outs), dim=-1)