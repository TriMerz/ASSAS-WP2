
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class ProbAttention(nn.Module):
    """
    Class used to implement Probabilistic Attention mechanism:
        - _prob_QK: Calculates the sampled Query-Key matrix)
        - _get_initial_context: Calculates the initial context for attention
    """
    def __init__(self,
                 mask_flag=True,
                 factor=5,
                 scale=None,
                 attention_dropout=0.1,
                 output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Randomly samples sample_k keys for each query
        Calculates an importance score for each query:
            - M = max_score - (sum_scores/L_K)
            - Higher = more informative

        Selects the n_top queries with highest scores
        Computes final attention only between selected queries and all keys
        Vantaggi:
            - Reduces complexity from O(L_Q * L_K) to O(n_top * L_K) while maintaining similar performance to full attention       
        """
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Calculate sample_k and n_top with safety checks
        sample_k = min(sample_k, L_K)  # Ensure sample_k doesn't exceed L_K
        n_top = min(n_top, L_Q)  # Ensure n_top doesn't exceed L_Q

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        
        # Safe sampling
        if L_K < sample_k:
            index_sample = torch.arange(L_K).unsqueeze(0).expand(L_Q, -1)
        else:
            index_sample = torch.randint(L_K, (L_Q, sample_k))
            
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        
        # Safe topk operation
        n_top = min(n_top, M.size(-1))  # Additional safety check
        if n_top < 1:
            n_top = 1
        M_top = M.topk(n_top, sorted=False)[1]

        # use reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Get the initial context based on the input values:
            - If mask_flag is False, returns the mean of the values
            - If mask_flag is True, returns the cumulative sum of the values
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Safe calculation of U_part and u
        U_part = self.factor * np.ceil(np.log(max(L_K, 1))).astype('int').item()
        u = self.factor * np.ceil(np.log(max(L_Q, 1))).astype('int').item()

        # Ensure minimum values and don't exceed sequence lengths
        U_part = max(min(U_part, L_K), 1)
        u = max(min(u, L_Q), 1)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Apply attention mask if provided
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            scores_top = scores_top.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax and dropout
        scores_top = self.dropout(F.softmax(scores_top, dim=-1))

        # Get the weighted sum of values
        context = torch.matmul(scores_top, values)
        
        # Restore original shape
        context = context.transpose(2, 1)

        return context

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
                 input_size,
                 embedding_dim,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 num_layers=3,
                 n_heads=8,
                 dropout=0.1):
        
        super(Autoencoder, self).__init__()
        self.window_size, self.n_features = input_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.n_heads = n_heads
        
        # Embeddings
        self.temporal_embedding = nn.Sequential(
            nn.Linear(15, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        self.feature_embedding = nn.Linear(self.n_features - 15, embedding_dim * 3 // 4)
        
        # Multi-scale attention
        self.attention_scales = nn.ModuleList([
            ProbAttention(True, factor=5, attention_dropout=dropout)
            for _ in range(3)  # 3 different scales
        ])
        
        # TCN layers
        self.tcn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2**i, padding='same', dilation=2**i),
                nn.GELU(),
                nn.BatchNorm1d(embedding_dim)
            ) for i in range(4)  # increasing receptive field
        ])
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([self._build_encoder_layer(embedding_dim, n_heads, dropout) for _ in range(num_layers)])
        
        # Decoder stack with skip connections
        self.decoder_layers = nn.ModuleList([self._build_decoder_layer(embedding_dim, n_heads, dropout) for _ in range(num_layers)])
        
        # Output projections
        self.temporal_output = nn.Linear(embedding_dim // 4, 2)
        self.feature_output = nn.Linear(embedding_dim * 3 // 4, self.n_features - 2)
        
    def _build_encoder_layer(self, dim, heads, dropout):
        return nn.ModuleDict({'attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout), dim, heads),
                              'mlp': nn.Sequential(nn.Linear(dim, dim * 4),
                                                   nn.GELU(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(dim * 4, dim)
                                                   ),
                               'tcn': nn.Sequential(nn.Conv1d(dim, dim * 2, kernel_size=5, padding='same'),
                                                    nn.GELU(),
                                                    nn.Conv1d(dim * 2, dim, kernel_size=5, padding='same')
                                                    ),
                               'norm1': nn.LayerNorm(dim),
                               'norm2': nn.LayerNorm(dim),
                               'norm3': nn.LayerNorm(dim)
                               })
        
    def _build_decoder_layer(self, dim, heads, dropout):
        return nn.ModuleDict({'cross_attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout), dim, heads),
                              'self_attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout), dim, heads),
                              'mlp': nn.Sequential(nn.Linear(dim, dim * 4),
                                                   nn.GELU(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(dim * 4, dim)
                                                   ),
                              'tcn': nn.Sequential(nn.Conv1d(dim, dim * 2, kernel_size=5, padding='same'),
                                                   nn.GELU(),
                                                   nn.Conv1d(dim * 2, dim, kernel_size=5, padding='same')
                                                   ),
                              'norm1': nn.LayerNorm(dim),
                              'norm2': nn.LayerNorm(dim),
                              'norm3': nn.LayerNorm(dim),
                              'norm4': nn.LayerNorm(dim)
                              })

    def _apply_multi_scale_attention(self, x):
        outputs = []
        # Reshape x to include head dimension before passing to attention
        B, L, D = x.shape
        x_reshaped = x.view(B, L, self.n_heads, D // self.n_heads)
        
        for attention in self.attention_scales:
            # Apply attention at different temporal scales
            scale_out = attention(x_reshaped, x_reshaped, x_reshaped, None)
            outputs.append(scale_out)
        
        # Average the outputs and reshape back
        output = torch.mean(torch.stack(outputs), dim=0)
        return output.view(B, L, D)

    def encode(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Split and embed data
        temporal_data = x[:, :, :15]
        feature_data = x[:, :, 15:]
        temporal_embedded = self.temporal_embedding(temporal_data)
        feature_embedded = self.feature_embedding(feature_data)
        x = torch.cat([temporal_embedded, feature_embedded], dim=-1)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder processing
        for layer in self.encoder_layers:
            # Multi-scale attention
            attn_out = self._apply_multi_scale_attention(x)
            x = layer['norm1'](x + attn_out)
            
            # MLP
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)
            
            # TCN
            tcn_out = layer['tcn'](x.transpose(-1, -2)).transpose(-1, -2)
            x = layer['norm3'](x + tcn_out)
            
            # Store skip connection
            skip_connections.append(x)
        
        return x, skip_connections

    def decode(self, embedding, skip_connections):
        """
        embedding shape: [batch_size, seq_len, embedding_dim]
        """
        # If embedding doesn't have sequence dimension, add it
        if len(embedding.shape) == 2:
            x = embedding.unsqueeze(1).repeat(1, self.window_size, 1)
        else:
            # If embedding already has sequence dimension, use as is
            x = embedding
        
        for i, layer in enumerate(self.decoder_layers):
            # Self attention
            self_attn = layer['self_attention'](x, x, x, None)
            x = layer['norm1'](x + self_attn)
            
            # Cross attention with encoder skip connection
            if i < len(skip_connections) and skip_connections[i] is not None:
                cross_attn = layer['cross_attention'](x, skip_connections[-i-1], skip_connections[-i-1], None)
                x = layer['norm2'](x + cross_attn)
            
            # MLP
            mlp_out = layer['mlp'](x)
            x = layer['norm3'](x + mlp_out)
            
            # TCN
            tcn_out = layer['tcn'](x.transpose(-1, -2)).transpose(-1, -2)
            x = layer['norm4'](x + tcn_out)
        
        # Split and project output
        temporal_out = self.temporal_output(x[:, :, :self.embedding_dim // 4])
        feature_out = self.feature_output(x[:, :, self.embedding_dim // 4:])
        
        return torch.cat([temporal_out, feature_out], dim=-1)

    def forward(self, x):
        embedding, skip_connections = self.encode(x)
        reconstruction = self.decode(embedding, skip_connections)
        return embedding, reconstruction