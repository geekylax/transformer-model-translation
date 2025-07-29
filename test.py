import pandas as pd 
import torch.nn as nn 
import torch
from config import *
from train import get_model,get_ds,run_validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
config = get_config()

# Fix sequence length to match the trained model
config["seq_len"] = 256  # The model was trained with seq_len=256

print("Loading dataset")
train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt= get_ds(config)

print("Loading model")

model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size())
model.to(device)
model_filename = get_weights_file_path(config, f"{8}")
print(model_filename)
print("Loading model state dict")
checkpoint = torch.load(model_filename,map_location=device)
# Load only the model state dict, not the entire checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
run_validation(model,test_dataloader,tokenizer_src,tokenizer_tgt,config,device,0,None,n_samples=10)



