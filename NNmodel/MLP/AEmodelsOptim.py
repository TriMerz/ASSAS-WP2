
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from transformers import TimeSeriesTransformerModel

"""
    Il modello ha come input un tensore di dimensione [batch_size, window_size, n_features]
    Attualmente le features sono 15 + 2233 = 2248

    Le prime 15 sono features temporali (timestamp) e le sue trasformazioni
    Le successive 2233 sono le features provenienti da ASTEC:

              |    1. Steam Partial-Pressure
              |    2. (2nd to last-3 (here only 2nd)) Non-Condensable Gas Partial-Pressure (here hydrogen)
    volumi di |    3. Void Fraction
    controllo |    4. Gas Temperature
              |    5. Liquid Temperature
              |    6. Void Fraction below the water level

     Junction |    7. Gas Velocity
              |    8. Liquis Speed

      Pumps   |    9. Pump Rotary speed
        
      Walls   |    10. Surface Temperature (S1)
              |    11. Surface Temperature (S2)

    Lo scopo dell'autoencoder è quello di comprimere le 2248 features in un embedding di dimensione ridotta (embedding_dim)
    mantendendo le informazioni più importanti.

    L'efficacia dell'autoencoder viene valutata tramite la capacità di ricostruire fedelmente le features originali.

    Una volta addestrato, l'autoencoder può essere utilizzato per comprimere i dati prima di passarliad una modello per forecasting.
"""


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale or 1./math.sqrt(64)  # Valore di default per d_keys
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Cache delle dimensioni
        B, H, L_Q, D = Q.shape
        L_K = K.shape[2]
        
        # Pre-allocazione della memoria
        K_expand = K.unsqueeze(-3)
        
        # Campionamento efficiente
        if L_K < sample_k:
            index_sample = torch.arange(L_K, device=Q.device)
            K_sample = K_expand[..., index_sample, :]
        else:
            index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
            K_sample = K_expand[..., index_sample, :]
        
        # Calcolo ottimizzato delle scores
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1))
        
        # Calcolo stabile di M
        M = Q_K_sample.max(-1)[0] - torch.logsumexp(Q_K_sample, dim=-1)
        
        M_top = torch.topk(M, min(n_top, L_Q), sorted=False)[1]
        
        # Gather ottimizzato
        Q_reduce = Q.gather(2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))
        
        return torch.matmul(Q_reduce, K.transpose(-2, -1)), M_top
    
class AttentionLayer(nn.Module):
    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        out = out.contiguous().view(B, L, -1)
        return self.out_projection(out)


class Autoencoder(nn.Module):
    def __init__(self,
                 n_features: int,
                 window_size: int = 20,
                 embedding_dim: int = 128,
                 device: str = 'cuda',
                 num_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        
        super(Autoencoder, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Manteniamo gli embedding separati per feature ma ottimizziamo l'implementazione
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Conv1d(1, embedding_dim, kernel_size=3, padding=1, device=device),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim)
            ).to(device) for _ in range(n_features)
        ])
        
        # Manteniamo le multiple scale di attention ma ottimizziamo
        self.attention_scales = nn.ModuleList([
            ProbAttention(
                True, 
                factor=5, 
                attention_dropout=dropout,
                device=device
            ).to(device) for _ in range(3)
        ])
        
        # TCN ottimizzato con dilatazioni multiple
        self.tcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    embedding_dim,
                    embedding_dim,
                    kernel_size=2**i,
                    padding='same',
                    dilation=2**i,
                    device=device
                ),
                nn.GELU(),
                nn.BatchNorm1d(embedding_dim, device=device)
            ).to(device) for i in range(4)
        ])
        
        # Ottimizzazione CUDA: pre-allocazione buffer
        self.register_buffer('attention_mask', torch.ones(window_size, window_size))
        
        # Encoder layers con ottimizzazioni CUDA
        self.encoder_layers = nn.ModuleList([
            self._build_encoder_layer(embedding_dim, n_heads, dropout).to(device)
            for _ in range(num_layers)
        ])
        
        # Decoder layers con ottimizzazioni CUDA
        self.decoder_layers = nn.ModuleList([
            self._build_decoder_layer(embedding_dim, n_heads, dropout).to(device)
            for _ in range(num_layers)
        ])
        
        # Output projections ottimizzati
        self.output_projections = nn.ModuleList([
            nn.Linear(embedding_dim, window_size, device=device)
            for _ in range(n_features)
        ])
        
        # Inizializzazione ottimizzata dei pesi
        self._init_weights()
        
    def _init_weights(self):
        # Inizializzazione ottimizzata per training stabile
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    @torch.cuda.amp.autocast('cuda')  # Ottimizzazione CUDA: mixed precision
    def _apply_multi_scale_attention(self, x):
        outputs = []
        B, F, D = x.shape
        x_reshaped = x.view(B, F, self.n_heads, -1)
        
        # Ottimizzazione CUDA: compute in parallel
        attention_outputs = [
            attention(x_reshaped, x_reshaped, x_reshaped, self.attention_mask) 
            for attention in self.attention_scales
        ]
        output = torch.stack(attention_outputs).mean(dim=0)
        return output.view(B, F, D)

    @torch.cuda.amp.autocast('cuda')
    def encode(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous()  # Ottimizzazione CUDA: ensure contiguous memory
        
        # Parallel feature processing
        feature_embeddings = []
        for i in range(self.n_features):
            feature_data = x[:, i, :]
            with torch.cuda.amp.autocast():
                embedding = self.feature_embeddings[i](feature_data)
            feature_embeddings.append(embedding)
        
        x = torch.stack(feature_embeddings, dim=1)
        
        skip_connections = []
        for layer in self.encoder_layers:
            # Ottimizzazione CUDA: gradient checkpointing
            if self.training:
                attn_out = torch.utils.checkpoint.checkpoint(
                    self._apply_multi_scale_attention, x
                )
            else:
                attn_out = self._apply_multi_scale_attention(x)
            
            x = layer['norm1'](x + attn_out)
            
            # Parallel attention computation
            self_attn = layer['self_attention'](x, x, x, self.attention_mask)
            cross_attn = layer['cross_feature_attention'](x, x, x, self.attention_mask)
            
            x = layer['norm2'](x + self_attn)
            x = layer['norm3'](x + cross_attn)
            
            # Parallel processing TCN and MLP
            x_trans = x.transpose(1, 2)
            tcn_out = layer['tcn'](x_trans).transpose(1, 2)
            mlp_out = layer['mlp'](x)
            x = layer['norm4'](x + mlp_out + tcn_out)
            
            # Optimized TCN processing
            with torch.cuda.amp.autocast():
                for tcn in self.tcn_layers:
                    x_trans = x.transpose(1, 2)
                    x = x + tcn(x_trans).transpose(1, 2)
            
            skip_connections.append(x)
        
        return x, skip_connections

    @torch.cuda.amp.autocast('cuda')
    def decode(self, embedding, skip_connections):
        x = embedding
        
        for i, layer in enumerate(self.decoder_layers):
            # Parallel attention computation
            self_attn = layer['self_attention'](x, x, x, self.attention_mask)
            x = layer['norm1'](x + self_attn)
            
            if i < len(skip_connections):
                cross_attn = layer['cross_attention'](
                    x, skip_connections[-i-1], skip_connections[-i-1], 
                    self.attention_mask
                )
                x = layer['norm2'](x + cross_attn)
            
            # Parallel processing
            with torch.cuda.amp.autocast('cuda'):
                x_trans = x.transpose(1, 2)
                tcn_out = layer['tcn'](x_trans).transpose(1, 2)
                mlp_out = layer['mlp'](x)
                x = layer['norm3'](x + mlp_out + tcn_out)
            
                for tcn in self.tcn_layers:
                    x_trans = x.transpose(1, 2)
                    x = x + tcn(x_trans).transpose(1, 2)
        
        # Parallel feature reconstruction
        reconstructions = []
        for i in range(self.n_features):
            feature_reconstruction = self.output_projections[i](x[:, i, :])
            reconstructions.append(feature_reconstruction)
        
        return torch.stack(reconstructions, dim=2)

    def forward(self, x):
        embedding, skip_connections = self.encode(x)
        reconstruction = self.decode(embedding, skip_connections)
        return embedding, reconstruction