import torch 
import torch.nn as nn
from model import Transformer
from config import get_config,get_weights_file_path
from train import get_model,get_ds,greedy_decode
import altair as alt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

config = get_config()

config["seq_len"] = 256  # The model was trained with seq_len=256

print("Loading dataset")
train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt= get_ds(config)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size())
model.to(device)
model_filename = get_weights_file_path(config, f"{8}")
checkpoint = torch.load(model_filename,map_location=device)
# Load only the model state dict, not the entire checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



def load_next_batch():
    # Load a sample batch from the test set
    batch = next(iter(test_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_tokens = [tokenizer_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [tokenizer_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # check that the batch size is 1
    assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

    model_out = greedy_decode(model, encoder_input, encoder_mask, config["seq_len"], tokenizer_tgt, config)
    
    return batch, encoder_input_tokens, decoder_input_tokens

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def get_attn_map(attn_type: str, layer: int, head: int):
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.layers[layer].cross_attention.attention_scores
    return attn[0, head].data

def attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        get_attn_map(attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(attn_type: str, layers: list[int], heads: list[int], row_tokens: list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)

# Run the attention visualization
batch, encoder_input_tokens, decoder_input_tokens = load_next_batch()
print(f'Source: {batch["src_text"][0]}')
print(f'Target: {batch["tgt_text"][0]}')
sentence_len = encoder_input_tokens.index("[PAD]")

layers = [0, 1, 2]
heads = [0, 1, 2, 3, 4, 5, 6, 7]

# Encoder Self-Attention
encoder_self_attn = get_all_attention_maps("encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))

# Decoder Self-Attention
decoder_self_attn = get_all_attention_maps("decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

# Encoder-Decoder Cross-Attention
encoder_decoder_attn = get_all_attention_maps("encoder-decoder", layers, heads, decoder_input_tokens, encoder_input_tokens, min(20, sentence_len))

encoder_self_attn.show()
decoder_self_attn.show()
encoder_decoder_attn.show()
encoder_self_attn.save("encoder_self_attn.html")
decoder_self_attn.save("decoder_self_attn.html")
encoder_decoder_attn.save("encoder_decoder_attn.html")
print("Attention maps generated successfully!")
print("You can display them using:")
print("encoder_self_attn.show()")
print("decoder_self_attn.show()")
print("encoder_decoder_attn.show()")





