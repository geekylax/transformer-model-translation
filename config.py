from pathlib import Path
import torch

def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 20,
        "seq_len": 256,
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "lr": 1e-4,
        "model_folder": "weights",
        "model_filename": "tam_model",
        "preload": None,
        "tokenizer_file_path": "tokenizer_en_ta.json",
        "lang_src": "en",
        "lang_tgt": "ta",
        "num_workers": 2,
        "pin_memory": True,
        "load_from_checkpoint": False,
        "summary_writer_dir": "runs/tam_model",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_filename = config["model_filename"]
    model_file_path = Path(model_folder) / f"{model_filename}_{epoch}.pth"
    return model_file_path