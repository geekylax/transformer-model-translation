from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def greedy_decode(model, src, src_mask, max_len, tokenizer_tgt, config):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.ones(1, 1).fill_(sos_idx).type(src.type()).to(src.device)
    while True:
        if decoder_input.size(1) >= max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src).to(src.device)
        decoder_output = model.decode(decoder_input, encoder_output, src_mask, decoder_mask)
        logits = model.project(decoder_output)[:, -1, :] / (config["d_model"] ** 0.5)
        probs = torch.softmax(logits, dim=-1)
        next_word = torch.argmax(probs, dim=-1).item()
        decoder_input = torch.cat([decoder_input, torch.ones(1, 1).type_as(src).fill_(next_word).to(src.device)], dim=1)
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model, val_ds, tokenizer_src, tokenizer_tgt, config, device, global_step, writer,n_samples=2000):
    model.eval()
    count = 0
    source_texts = []
    target_texts = []
    predictions = []
    console_width = 200
    batch_iterator = tqdm(val_ds, desc="Running Validation", leave=False)
    
    with torch.no_grad():
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Encoder input size is not equal to sequence length"
            model_out = greedy_decode(model, encoder_input, encoder_mask, max_len=config["seq_len"], tokenizer_tgt=tokenizer_tgt, config=config)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.cpu().numpy())
            model_out_text = model_out_text.replace("<bos>", "").replace("<eos>", "")
            model_out_text = model_out_text.replace("[PAD]", "").replace("[UNK]", "")
            model_out_text = model_out_text.replace("[SOS]", "").replace("[EOS]", "")
            model_out_text = model_out_text.replace("[PAD]", "").replace("[UNK]", "")
            source_texts.append(source_text)
            target_texts.append(target_text)
            predictions.append(model_out_text)
            count += 1
            print(f"Processed {count} samples")
            print(f"Source text: {source_text}")
            print(f"Target text: {target_text}")
            print(f"Model output: {model_out_text}")
            print("-" * console_width)
            if count == n_samples:
                break
            
            

def get_all_sentences(dataset, lang):
    """Generator that yields sentences from dataset for tokenizer training"""
    for item in dataset:
        yield item[lang]  # Access 'en' or 'ta' directly

def get_or_build_tokenizer(config, dataset, name):
    tokenizer_file_path = Path(config["tokenizer_file_path"]) / f"{name}.json"
    
    if not Path.exists(tokenizer_file_path):
        tokenizer_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(dataset, name), trainer)
        
        tokenizer.save(str(tokenizer_file_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_file_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("Hemanth-thunder/en_ta", split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, "en")
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, "ta")

    ## keep 90% of the data
    ds_raw = ds_raw.train_test_split(test_size=0.1)

    ## get all the sentences

    train_ds = BilingualDataset(ds_raw["train"], tokenizer_src, tokenizer_tgt, "en", "ta", config["seq_len"])
    test_ds = BilingualDataset(ds_raw["test"], tokenizer_src, tokenizer_tgt, "en", "ta", config["seq_len"])
    max_len_src= 0 
    max_len_tgt= 0
    
    for item in tqdm(ds_raw["train"],desc="Calculating max lengths"):
        src_ids = tokenizer_src.encode(item["en"]).ids
        tgt_ids = tokenizer_tgt.encode(item["ta"]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Maximum length of source sentences: {max_len_src}")
    print(f"Maximum length of target sentences: {max_len_tgt}")
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], d_model=config["d_model"], N=config["N"], num_heads=config["num_heads"], d_ff=config["d_ff"], dropout=config["dropout"])
    return model

def train_model(config=None):
    ## get the device
    def get_device():
        """Get the best available device"""
        if torch.cuda.is_available():
            print("Using CUDA GPU")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Mac GPU (MPS)")
            return torch.device("mps")
        else:
            print("Using CPU")
            return torch.device("cpu")

    device = get_device()
    print(f"Using device: {device}")
    
    if config is None:
        config = get_config()
        
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config["summary_writer_dir"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)

    init_epoch = 0
    if config["preload"]:
        model_filenam = get_weights_file_path(config, config["preload"])
        state_dict = torch.load(model_filenam)
        init_epoch = state_dict["epoch"]+1
        optimizer.load_state_dict(state_dict["optimizer"])
        global_step = state_dict["global_step"]
        print(f"Resuming training from epoch {init_epoch} and global step {global_step}")
    else:
        global_step = 0
        init_epoch = 0
        print("Initializing a new training run")
        
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

    for epoch in range(init_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)
            label = batch["label"].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            logits = model.project(decoder_output)
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            optimizer.zero_grad()

            loss.backward()
            batch_iterator.set_postfix({"loss": loss.item()})
            writer.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1
            optimizer.step()
            writer.flush()
            global_step += 1
            run_validation(model, test_dataloader, tokenizer_src, tokenizer_tgt, config, device,  global_step, writer, n_samples=2000)
            
        model_filename = get_weights_file_path(config, f"{epoch+1}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)
        print(f"Saved checkpoint to {model_filename}")


if __name__ == "__main__":
    config = get_config()
    train_model(config)