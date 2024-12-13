
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
        self.first_forward = True

    def _prob_QK(self, Q, K, sample_k, n_top):
        should_print = self.first_forward
        if should_print:
            print("\n=== _prob_QK Debug ===")
            print(f"Q shape: {Q.shape}")
            print(f"K shape: {K.shape}")
            print(f"sample_k: {sample_k}, n_top: {n_top}")
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
        if should_print:
            print(f"K_expand shape: {K_expand.shape}")
        
        # Safe sampling
        if L_K < sample_k:
            index_sample = torch.arange(L_K).unsqueeze(0).expand(L_Q, -1)
        else:
            index_sample = torch.randint(L_K, (L_Q, sample_k))
            
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        if should_print:
            print(f"K_sample shape: {K_sample.shape}")
        
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        if should_print:
            print(f"Q_K_sample shape: {Q_K_sample.shape}")

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

        if should_print:
            print(f"Final Q_K shape: {Q_K.shape}")
            print("=== End _prob_QK Debug ===\n")

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        should_print = self.first_forward
        if should_print:
            print("\n=== _get_initial_context Debug ===")
            print(f"V shape: {V.shape}")
            print(f"L_Q: {L_Q}")
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

        if should_print:
            print(f"Context shape: {contex.shape}")
            print("=== End _get_initial_context Debug ===\n")

        return contex

    def forward(self, queries, keys, values, attn_mask):
        should_print = self.first_forward
        self.first_forward = False
        
        if should_print:
            print("\n=== ProbAttention Forward Debug ===")
            print(f"Input shapes:")
            print(f"- queries: {queries.shape}")
            print(f"- keys: {keys.shape}")
            print(f"- values: {values.shape}")

        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        if should_print:
            print(f"After transpose shapes:")
            print(f"- queries: {queries.shape}")
            print(f"- keys: {keys.shape}")
            print(f"- values: {values.shape}")

        U_part = self.factor * np.ceil(np.log(max(L_K, 1))).astype('int').item()
        u = self.factor * np.ceil(np.log(max(L_Q, 1))).astype('int').item()

        U_part = max(min(U_part, L_K), 1)
        u = max(min(u, L_Q), 1)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        if should_print:
            print(f"scores_top shape: {scores_top.shape}")

        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            scores_top = scores_top.masked_fill(attn_mask == 0, float('-inf'))

        scores_top = self.dropout(F.softmax(scores_top, dim=-1))
        if should_print:
            print(f"After softmax shape: {scores_top.shape}")

        context = torch.matmul(scores_top, values)
        if should_print:
            print(f"Context shape before transpose: {context.shape}")
        
        context = context.transpose(2, 1)
        if should_print:
            print(f"Final output shape: {context.shape}")
            print("=== End ProbAttention Forward ===\n")
        
        return context

class AttentionLayer(nn.Module):
    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Assicuriamoci che le dimensioni siano divisibili
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        self.d_keys = d_keys
        self.d_values = d_values

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)  # d_model = n_heads * d_keys
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.first_forward = True

    def forward(self, queries, keys, values, attn_mask):
        should_print = self.first_forward
        self.first_forward = False

        if should_print:
            print("\n=== AttentionLayer Debug ===")
            print(f"Input shapes:")
            print(f"- queries: {queries.shape}")
            print(f"- keys: {keys.shape}")
            print(f"- values: {values.shape}")
        
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
        if should_print:
            print(f"After query projection: {queries.shape}")
            print(f"After key projection: {keys.shape}")
            print(f"After value projection: {values.shape}")
        
        head_dim = queries.shape[-1] // H
        queries = queries.view(B, L, H, head_dim).transpose(1, 2)
        keys = keys.view(B, S, H, head_dim).transpose(1, 2)
        values = values.view(B, S, H, head_dim).transpose(1, 2)
        
        if should_print:
            print(f"After queries reshape: {queries.shape}")
            print(f"After keys reshape: {keys.shape}")
            print(f"After values reshape: {values.shape}")
        
        out = self.inner_attention(queries, keys, values, attn_mask)
        
        if should_print:
            print(f"After attention: {out.shape}")
        
        out = out.transpose(1, 2).contiguous()
        if should_print:
            print(f"After transpose: {out.shape}")
        
        out = out.view(B, L, -1)
        if should_print:
            print(f"After view: {out.shape}")
        
        out = self.out_projection(out)
        if should_print:
            print(f"Final output: {out.shape}")
            print("=== End AttentionLayer Debug ===\n")
        
        return out



# Hybrid Approach - create an embedding for each feature and process them separately
class Autoencoder(nn.Module):
    def __init__(self,
                 n_features: int,
                 window_size: int = 20,
                 embedding_dim: int = 128,
                 num_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        super(Autoencoder, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.first_forward = True 
        self.device = device

        """ 
            Encoding Process:
            1. Input Shape: [batch, window_size=20, n_features=2248]
                - Batch: numero di campioni
                - Window_size: sequenza temporale di 20 timestep
                - N_features: 2248 features (15 temporali + 2233 ASTEC)

            2. Transpose: [batch, n_features, window_size]
                - Riorganizza per processare ogni feature separatamente

            3. Per ogni feature i (0 -> 2247):
                a. Estrai singola feature: x[:, i, :] → [batch, window_size]
                    Es: se i=0 → Steam Partial-Pressure per tutti i batch nei 20 timestep
                    
                b. Aggiungi dimensione canale: unsqueeze(1) → [batch, 1, window_size]
                
                c. PRIMA CONVOLUZIONE 1D: (in_channels=1, out_channels=hidden_dim, kernel=3)
                    - Scorre sulla sequenza temporale guardando 3 timestep alla volta
                    - Cerca pattern temporali base (es: aumenti, diminuzioni)
                    - Output: [batch, hidden_dim, window_size]
                
                d. SECONDA CONVOLUZIONE 1D: (in_channels=hidden_dim, out_channels=embedding_dim)
                    - Combina i pattern base in pattern più complessi
                    - Output: [batch, embedding_dim, window_size]
                
                e. Feature Embedding:
                    - Flatten: [batch, embedding_dim * window_size]
                    - Proiezione lineare → [batch, embedding_dim]
                    - Ogni feature ora ha un embedding di dimensione fissa

            4. Stack degli embedding: [batch, n_features, embedding_dim]
                - Combina tutti gli embedding delle feature

            5. Cross-Feature Processing:
                - Attention e MLP per catturare relazioni tra feature diverse
                - Mantiene skip connections per il decoder

            Il processo è feature-specific: ogni feature viene processata indipendentemente 
            attraverso le convoluzioni prima che avvenga qualsiasi interazione tra feature diverse.
        """

        # 1. Feature-specific processing (temporal convolution)
        """
            Vedi 3.c, 3.d
        """
        hidden_dim = self.embedding_dim // 2
        self.temporal_processing = nn.ModuleList([nn.Sequential(nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),    # Prima conv mantiene dimensione temporale
                                                                nn.GELU(),
                                                                nn.BatchNorm1d(hidden_dim),
                                                                nn.Conv1d(hidden_dim, embedding_dim, kernel_size=3, padding=1),    # Seconda conv mantiene dimensione temporale
                                                                nn.GELU(),
                                                                nn.BatchNorm1d(embedding_dim)
                                                                # Rimuoviamo qualsiasi layer che modifica la dimensione temporale
                                                                ) for _ in range(n_features)
                                                                ])
        
        # Dimensione dell'input dopo temporal processing
        temporal_output_size = window_size * embedding_dim  # 20 * embedding_dim = 2560

        # 2. Feature-specific embeddings
        self.feature_embeddings = nn.ModuleList([nn.Sequential(nn.Flatten(),    # [batch, embedding_dim, window_size] -> [batch, embedding_dim * window_size]
                                                               nn.Linear(temporal_output_size, embedding_dim),    # [batch, 2560] -> [batch, 128]
                                                               nn.LayerNorm(embedding_dim),
                                                               nn.GELU(),
                                                               nn.Dropout(dropout)
                                                               ) for _ in range(n_features)
                                                               ])
        
        # 3. Cross-Feature Interaction Module
        self.cross_feature_layers = nn.ModuleList([nn.ModuleDict({'attention': AttentionLayer(ProbAttention(True, factor=5, attention_dropout=dropout),
                                                                                              embedding_dim,
                                                                                              n_heads
                                                                                              ),
                                                                  'mlp': nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 4),
                                                                                       nn.GELU(),
                                                                                       nn.Dropout(dropout),
                                                                                       nn.Linear(embedding_dim * 4, embedding_dim)
                                                                                       ),
                                                                  'norm1': nn.LayerNorm(embedding_dim),
                                                                  'norm2': nn.LayerNorm(embedding_dim)
                                                                  }) for _ in range(num_layers)
                                                                  ])
        
        # 4. Feature-specific processing post-interaction
        self.feature_processors = nn.ModuleList([nn.ModuleList([self._build_feature_processor(embedding_dim, n_heads, dropout)
                                                                for _ in range(num_layers)]) for _ in range(n_features)
                                                                ])
    
        # 5. Multi-scale attention per ogni feature
        self.feature_attention = nn.ModuleList([ProbAttention(True, factor=5, attention_dropout=self.dropout)
                                               for _ in range(self.n_features)  # 3 diverse scale temporali
                                               ])
        
        # 6. Feature-specific TCN
        self.feature_tcn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Conv1d(1, 1, kernel_size=2**i, padding='same', dilation=2**i),
                                                                       nn.GELU(),
                                                                       nn.BatchNorm1d(1)
                                                                       ) for i in range(4)
                                                                       ]) for _ in range(n_features)
                                                                       ])
        
        # Registrazione del buffer SENZA .to(device)
        self.register_buffer('attention_mask', torch.ones(window_size, window_size))
        
        # 7. Encoder layers con architettura avanzata
        self.feature_encoder = nn.ModuleList([nn.ModuleList([self._build_encoder_layer(self.embedding_dim, self.n_heads, self.dropout)
                                             for _ in range(self.num_layers)]) for _ in range(self.n_features)
                                             ])
        
        # 8. Decoder layers con skip connections e cross-attention
        self.feature_decoder = nn.ModuleList([nn.ModuleList([self._build_decoder_layer(self.embedding_dim, self.n_heads, self.dropout)
                                             for _ in range(self.num_layers)]) for _ in range(self.n_features)
                                             ])
        
        # 9. Output projection per feature
        self.output_projections = nn.ModuleList([nn.Linear(self.embedding_dim, self.window_size)
                                                 for _ in range(self.n_features)
                                                 ])
        
        # Inizializzazione dei pesi
        self._init_weights()
        # UNICO .to(device) per l'intero modello
        self.to(device)
        
    def _build_encoder_layer(self, dim, heads, dropout):
        return nn.ModuleDict({'self_attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout),
                                                               dim,
                                                               heads),
                              'cross_feature_attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout),
                                                                        dim,
                                                                        heads),
                              'mlp': nn.Sequential(nn.Linear(dim, dim * 4),
                                                   nn.GELU(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(dim * 4, dim)),
                              'tcn': nn.Sequential(nn.Conv1d(dim, dim * 2, kernel_size=5, padding='same'),
                                                   nn.GELU(),
                                                   nn.Conv1d(dim * 2, dim, kernel_size=5, padding='same')),
                              'norm1': nn.LayerNorm(dim),
                              'norm2': nn.LayerNorm(dim),
                              'norm3': nn.LayerNorm(dim),
                              'norm4': nn.LayerNorm(dim)
                              })
        
    def _build_decoder_layer(self, dim, heads, dropout):
        return nn.ModuleDict({'self_attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout),
                                                               dim,
                                                               heads),
                              'cross_attention': AttentionLayer(ProbAttention(True, 5, attention_dropout=dropout),
                                                                dim,
                                                                heads),
                              'mlp': nn.Sequential(nn.Linear(dim, dim * 4),
                                                   nn.GELU(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(dim * 4, dim)),
                              'tcn': nn.Sequential(nn.Conv1d(dim, dim * 2, kernel_size=5, padding='same'),
                                                   nn.GELU(),
                                                                nn.Conv1d(dim * 2, dim, kernel_size=5, padding='same')),
                              'norm1': nn.LayerNorm(dim),
                              'norm2': nn.LayerNorm(dim),
                              'norm3': nn.LayerNorm(dim),
                              'norm4': nn.LayerNorm(dim)
                              })
    
    def _build_feature_processor(self, dim, heads, dropout):
        return nn.Sequential(nn.Linear(dim, dim * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 2, dim),
                            nn.LayerNorm(dim)
                            )
    
    def _apply_multi_scale_attention(self, x):
        # x shape: [batch_size, n_features, embedding_dim]
        outputs = []
        B, F, D = x.shape
        x_reshaped = x.view(B, F, self.n_heads, -1)     # Applica attention a 3 scale diverse
                                                        #    [batch, 2248, embedding_dim]
        
        for attention in self.attention_scales:
            scale_out = attention(x_reshaped, x_reshaped, x_reshaped, None)
            outputs.append(scale_out)
        
        output = torch.mean(torch.stack(outputs), dim=0)
        return output.view(B, F, D)

    def encode(self, x):
        # Controllo se è la prima iterazione
        should_print = self.first_forward
        self.first_forward = False
        
        if should_print:
            print("\n=== Starting Encoding Process ===")
            print(f"Initial input shape: {x.shape}")
        
        x = x.to(self.device)
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        
        feature_embeddings = []
        for i in range(self.n_features):
            if should_print and i == 0:
                print(f"\n=== Processing Feature {i} ===")
                print(f"1. Feature data shape: {x[:, i, :].unsqueeze(1).shape}")
                
            feature_data = x[:, i, :].unsqueeze(1)
            
            # Temporal processing con debug solo per la prima feature nella prima iterazione
            if should_print and i == 0:
                print("\n2. Temporal Processing:")
                temporal_features = feature_data
                for idx, layer in enumerate(self.temporal_processing[i]):
                    temporal_features = layer(temporal_features)
                    print(f"   After layer {idx}: {temporal_features.shape}")
                print(f"Final temporal features shape: {temporal_features.shape}")
            else:
                temporal_features = self.temporal_processing[i](feature_data)
            
            # Feature embedding con debug solo per la prima feature nella prima iterazione
            if should_print and i == 0:
                print("\n3. Feature Embedding:")
                embedding = temporal_features
                for idx, layer in enumerate(self.feature_embeddings[i]):
                    print(f"   Before layer {idx}: {embedding.shape}")
                    embedding = layer(embedding)
                    print(f"   After layer {idx}: {embedding.shape}")
                print(f"\nFinal embedding shape for feature {i}: {embedding.shape}")
            else:
                embedding = self.feature_embeddings[i](temporal_features)
            
            feature_embeddings.append(embedding)
        
        embeddings = torch.stack(feature_embeddings, dim=1)
        
        if should_print:
            print(f"\nFinal stacked embeddings shape: {embeddings.shape}")
            print("\n=== Starting Cross-Feature Processing ===")
        
        skip_connections = []
        
        # Cross-Feature Processing con debug solo nella prima iterazione
        for layer_idx, layer in enumerate(self.cross_feature_layers):
            if should_print:
                print(f"\nProcessing cross-feature layer {layer_idx}")
                print(f"Input shape: {embeddings.shape}")
            
            attn_out = layer['attention'](embeddings, embeddings, embeddings, None)
            embeddings = layer['norm1'](embeddings + attn_out)
            mlp_out = layer['mlp'](embeddings)
            embeddings = layer['norm2'](embeddings + mlp_out)
            
            if should_print:
                print(f"After attention shape: {attn_out.shape}")
                print(f"After norm1 shape: {embeddings.shape}")
                print(f"After MLP shape: {mlp_out.shape}")
                print(f"After norm2 shape: {embeddings.shape}")
            
            skip_connections.append(embeddings)
        
        # Final processing con debug solo nella prima iterazione
        processed_embeddings = []
        if should_print:
            print("\n=== Final Feature Processing ===")
        
        for i in range(self.n_features):
            feature_emb = embeddings[:, i, :]
            
            if should_print and i == 0:
                print(f"\nProcessing final feature {i}")
                print(f"Initial shape: {feature_emb.shape}")
            
            for layer_idx, layer in enumerate(self.feature_processors[i]):
                feature_emb = layer(feature_emb)
                if should_print and i == 0:
                    print(f"After processor layer {layer_idx}: {feature_emb.shape}")
            
            processed_embeddings.append(feature_emb)
        
        if should_print:
            print("\n=== Encoding Complete ===")
        
        return processed_embeddings, skip_connections
    
    def decode(self, embeddings, skip_connections):
        reconstructions = []
        skip_connections = [skip.to(self.device) for skip in skip_connections]    # Assicuriamo che skip_connections siano sul device corretto
        for i in range(self.n_features):        # Reverse the feature-specific processing
            embedding = embeddings[i].to(self.device)
            
            # Use skip connections
            for skip in reversed(skip_connections):
                embedding = embedding + skip[:, i, :]
            
            # Project back to original space
            reconstruction = self.output_projections[i](embedding)
            reconstructions.append(reconstruction)
        
        # Combine reconstructions
        return torch.stack(reconstructions, dim=2)  # [batch, window_size, n_features]

    def forward(self, x):
        # x shape: [batch_size, window_size, n_features]
        embedding, skip_connections = self.encode(x)
        reconstruction = self.decode(embedding, skip_connections)
        return embedding, reconstruction