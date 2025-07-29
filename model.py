import torch 
import torch.nn as nn 
import math 

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000,dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # ✅ Create temporary tensor (no self.anything)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        #Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])    
    
class LayerNormalization(nn.Module):
    def __init__(self, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiply
        self.bias = nn.Parameter(torch.zeros(1)) # add

    def forward(self, x):
        x = x.float()
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # batch, seq_len, d_model -> batch, seq_len, d_ff
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == self.d_model, "d_model must be divisible by num_heads"

        
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        if dropout is not None:
            attn_scores = dropout(attn_scores)
        
        return torch.matmul(attn_scores, value), attn_scores # Return both output and attention scores

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, d_model   )
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head_dim * self.num_heads)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_1 = ResidualConnection(dropout)
        self.residual_2 = ResidualConnection(dropout)

    def forward(self, x, mask=None):
        # Apply self-attention with residual connection
        x = self.residual_1(x, lambda y: self.self_attention(y, y, y, mask))
        # Apply feed-forward with residual connection  
        x = self.residual_2(x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention:MultiHeadAttention, cross_attention:MultiHeadAttention, feed_forward:FeedForwardBlock, dropout:float=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_1 = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_1[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_1[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_1[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int=512, N:int=6, num_heads:int=8, d_ff:int=2048, dropout:float=0.1):
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = nn.ModuleList()
    decoder_blocks = nn.ModuleList()

    for _ in range(N):
        # EncoderBlock creates its own attention and feed-forward internally
        encoder_blocks.append(EncoderBlock(d_model, num_heads, d_ff, dropout))

    for _ in range(N):
        # DecoderBlock needs pre-created components
        decoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout))

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

# # initlaise the parameters
# transformer = build_transformer(src_vocab_size=10000, tgt_vocab_size=10000, src_seq_len=100, tgt_seq_len=100, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1)

# # print the model
# print(transformer)