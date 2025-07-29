import modal
from pathlib import Path

# Modal setup
app = modal.App("transformer-model-training")
volume = modal.Volume.from_name("transformer-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol/data"

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "datasets>=2.14.0", 
        "tokenizers>=0.13.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "numpy<2.0.0",  # Use NumPy 1.x for TensorBoard compatibility
    ])
)

# Add ALL your local files to the container
image = image.add_local_dir(
    Path(__file__).parent,
    remote_path="/root/src"
)

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="H100",  # 🚀 H100 GPU - Maximum performance
    timeout=2600,  # 60 minutes for complete training (increased from 20 min)
    cpu=8,  # More CPU cores for H100
    memory=32768,  # 32GB RAM for H100
)
def train_transformer_on_modal():
    """
    OPTIMIZED transformer training on H100 GPU.
    Uses 1 lakh (100K) samples for faster training within budget.
    """
    import sys
    import os
    
    # Add your source files to Python path
    sys.path.insert(0, "/root/src")
    
    # Import your original helper functions
    from train import get_all_sentences, get_or_build_tokenizer, get_ds, get_model
    from config import get_config, get_weights_file_path
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    
    print("🚀 Starting OPTIMIZED transformer training on H100 GPU...")
    print(f"Available files: {os.listdir('/root/src')}")
    
    # Get your original config
    config = get_config()
    
    # Update paths to use Modal volume for persistent storage
    config["model_folder"] = f"{VOLUME_PATH}/weights"
    config["summary_writer_dir"] = f"{VOLUME_PATH}/runs/tam_model"
    config["tokenizer_file_path"] = f"{VOLUME_PATH}/tokenizer_en_ta.json"
    
    # 🚀 OPTIMIZED SETTINGS for H100 with 1 lakh data
    config["batch_size"] = 128        # Large batch for H100 (80GB VRAM)
    config["seq_len"] = 256           # Shorter sequences for speed
    config["num_epochs"] = 8          # More epochs since dataset is smaller
    config["lr"] = 2e-4               # Higher LR for faster convergence
    config["dataset_limit"] = 100000  # 🎯 1 lakh samples only
    
    # Keep your model architecture
    config["d_model"] = 512           # Your original model size
    config["N"] = 6                   # 6 transformer layers
    config["num_heads"] = 8           # 8 attention heads
    config["d_ff"] = 2048             # Feed-forward dimension
    
    print(f"🚀 H100 OPTIMIZED CONFIG (1 Lakh Dataset):")
    print(f"  - GPU: H100 (80GB VRAM) - MAXIMUM POWER!")
    print(f"  - Dataset: 100K samples (reduced from 285K)")
    print(f"  - Target time: 15-20 minutes")
    print(f"  - Estimated cost: $2-3")
    print(f"  - Model folder: {config['model_folder']}")
    print(f"  - TensorBoard logs: {config['summary_writer_dir']}")
    print(f"  - Tokenizers: {config['tokenizer_file_path']}")
    print(f"  - Batch size: {config['batch_size']} 🚀 (H100 optimized)")
    print(f"  - Sequence length: {config['seq_len']} (speed optimized)")
    print(f"  - Epochs: {config['num_epochs']} (more epochs, smaller dataset)")
    print(f"  - Learning rate: {config['lr']} (faster convergence)")
    print(f"  - Expected time per epoch: ~2-3 minutes")
    print(f"  - Model: d_model={config['d_model']}, layers={config['N']}, heads={config['num_heads']}")
    
    # Create directories
    os.makedirs(config["model_folder"], exist_ok=True)
    os.makedirs(config["summary_writer_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["tokenizer_file_path"]), exist_ok=True)
    
    # Run your original training logic (optimized for H100)
    try:
        print("🎯 Starting H100 OPTIMIZED training...")
        
        # Custom dataset loading with filtering
        def get_filtered_ds(config):
            """Load dataset with filtering for sequence length and sample count"""
            from datasets import load_dataset, Dataset
            from torch.utils.data import DataLoader
            from dataset import BilingualDataset  # ✅ Add missing import
            
            print("📊 Loading and filtering dataset...")
            ds_raw = load_dataset("Hemanth-thunder/en_ta", split="train")
            
            # First limit to 1 lakh samples for speed
            if len(ds_raw) > config["dataset_limit"]:
                ds_raw = ds_raw.select(range(config["dataset_limit"]))
                print(f"📊 Limited to {config['dataset_limit']:,} samples")
            
            tokenizer_src = get_or_build_tokenizer(config, ds_raw, "en")
            tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, "ta")
            
            # Filter out sequences longer than seq_len
            print(f"🔍 Filtering sequences longer than {config['seq_len']} tokens...")
            filtered_data = []
            max_len_src = 0
            max_len_tgt = 0
            
            for item in ds_raw:
                src_ids = tokenizer_src.encode(item["en"]).ids
                tgt_ids = tokenizer_tgt.encode(item["ta"]).ids
                
                # Keep only if both sequences fit within seq_len (with some buffer for special tokens)
                if len(src_ids) <= (config["seq_len"] - 2) and len(tgt_ids) <= (config["seq_len"] - 2):
                    filtered_data.append(item)
                    max_len_src = max(max_len_src, len(src_ids))
                    max_len_tgt = max(max_len_tgt, len(tgt_ids))
            
            print(f"✅ Filtered dataset: {len(filtered_data):,} samples (from {len(ds_raw):,})")
            print(f"📏 Max source length: {max_len_src}")
            print(f"📏 Max target length: {max_len_tgt}")
            
            # Convert back to dataset format
            filtered_dataset = Dataset.from_list(filtered_data)
            
            # Split into train/test
            ds_split = filtered_dataset.train_test_split(test_size=0.1)
            
            # Create BilingualDatasets
            train_ds = BilingualDataset(ds_split["train"], tokenizer_src, tokenizer_tgt, "en", "ta", config["seq_len"])
            test_ds = BilingualDataset(ds_split["test"], tokenizer_src, tokenizer_tgt, "en", "ta", config["seq_len"])
            
            # Create dataloaders
            train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
            test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)
            
            return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt
        
        # Your original get_device function
        def get_device():
            if torch.cuda.is_available():
                print("✅ Using H100 GPU - MAXIMUM POWER!")
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                print("Using Mac GPU (MPS)")
                return torch.device("mps")
            else:
                print("Using CPU")
                return torch.device("cpu")

        device = get_device()
        print(f"🚀 Device: {device}")
        
        # Load FILTERED dataset (1 lakh + length filtering)
        train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_filtered_ds(config)
        
        # Build model
        print("🏗️ Building transformer model...")
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params:,}")
        print(f"📊 Source vocabulary: {tokenizer_src.get_vocab_size():,}")
        print(f"📊 Target vocabulary: {tokenizer_tgt.get_vocab_size():,}")
        
        # Setup training components
        writer = SummaryWriter(config["summary_writer_dir"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)

        # 🔄 CHECKPOINT RESUMPTION LOGIC
        init_epoch = 0
        global_step = 0
        
        # Check for existing checkpoints
        import glob
        checkpoint_pattern = f"{config['model_folder']}/tam_model_*.pth"
        existing_checkpoints = glob.glob(checkpoint_pattern)
        
        if existing_checkpoints:
            # Find the latest checkpoint
            latest_checkpoint = max(existing_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            epoch_num = int(latest_checkpoint.split('_')[-1].split('.')[0])
            
            print(f"🔄 Found existing checkpoint: {latest_checkpoint}")
            print(f"📊 Resuming from epoch {epoch_num}")
            
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                init_epoch = checkpoint.get("epoch", epoch_num - 1) + 1
                global_step = checkpoint.get("global_step", 0)
                
                print(f"✅ Successfully loaded checkpoint")
                print(f"🔄 Resuming from epoch {init_epoch}, global step {global_step}")
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                print("🆕 Starting fresh training instead")
                init_epoch = 0
                global_step = 0
        else:
            print("🆕 No existing checkpoints found - starting fresh training")
            
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
        
        print(f"\n🚀 STARTING TRAINING FROM EPOCH {init_epoch + 1}")
        print(f"🎯 Will train epochs {init_epoch + 1} to {config['num_epochs']}")
        print("=" * 50)
        
        # Your original training loop with performance optimizations
        for epoch in range(init_epoch, config["num_epochs"]):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            print(f"\n📈 EPOCH {epoch+1}/{config['num_epochs']}")
            batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
            
            for batch in batch_iterator:
                encoder_input = batch["encoder_input"].to(device, non_blocking=True)
                decoder_input = batch["decoder_input"].to(device, non_blocking=True)
                encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
                decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                logits = model.project(decoder_output)
                
                loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
                optimizer.zero_grad()

                loss.backward()
                batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                writer.add_scalar("train_loss", loss.item(), global_step)
                writer.add_scalar("epoch", epoch + 1, global_step)  # Add epoch to logs
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                global_step += 1
                
                # Log every 100 steps
                if global_step % 100 == 0:
                    writer.flush()
                
            avg_loss = epoch_loss / batch_count
            print(f"✅ Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
            
            # Log epoch summary to TensorBoard
            writer.add_scalar("epoch_loss", avg_loss, epoch + 1)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch + 1)
                
            # Save checkpoint with proper naming: tam_model_1.pth, tam_model_2.pth, etc.
            model_filename = f"{config['model_folder']}/tam_model_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "loss": avg_loss,
                "config": config
            }, model_filename)
            print(f"💾 Saved: tam_model_{epoch+1}.pth")
            
            # Commit to Modal volume after each epoch
            volume.commit()
            print(f"☁️  Synced to Modal volume")
            
        writer.close()
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📊 Final epoch: {config['num_epochs']}")
        print(f"📊 Total steps: {global_step}")
        print("=" * 50)
        
        return {
            "status": "success",
            "message": "Training completed and saved to Modal volume",
            "model_folder": config["model_folder"],
            "tensorboard_logs": config["summary_writer_dir"]
        }
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"Training failed: {str(e)}"
        }

@app.local_entrypoint()
def main():
    """
    🚀 H100 OPTIMIZED TRAINING with FILTERED DATASET
    - 1 lakh (100K) samples maximum
    - Sequences filtered to ≤256 tokens
    - Optimized for H100 performance
    - Fast and efficient training
    """
    print("🚀 LAUNCHING H100 OPTIMIZED TRANSFORMER TRAINING")
    print("=" * 60)
    print("🎯 H100 TRAINING SPECIFICATIONS:")
    print("   🔥 GPU: H100 (80GB VRAM) - THE ULTIMATE!")
    print("   📦 Batch size: 128 (H100 optimized)")
    print("   📏 Sequence length: ≤256 tokens (filtered)")
    print("   🔄 Epochs: 8 (optimized for smaller dataset)")
    print("   📈 Dataset: ≤100K samples (filtered + limited)")
    print("   🔍 Filter: Removes sequences >256 tokens")
    print("   ⏱️  Estimated time: 15-20 minutes")
    print("   💰 Estimated cost: $2-3")
    print("   🎯 Quality: Fast, clean training!")
    print("=" * 60)
    print()
    
    print("📤 Uploading your files to Modal H100 server...")
    print("   ├── train.py (your training logic)")
    print("   ├── model.py (transformer architecture)") 
    print("   ├── dataset.py (data processing)")
    print("   ├── config.py (hyperparameters)")
    print("   └── All other project files")
    print()
    
    # Run training on Modal's H100 GPU servers
    print("🚀 Starting H100 training with filtered dataset...")
    result = train_transformer_on_modal.remote()
    
    if result["status"] == "success":
        print("\n" + "🎉" * 20)
        print("🚀 H100 TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print(f"✅ {result['message']}")
        print(f"📂 Models saved to: {result['model_folder']}")
        print(f"📊 TensorBoard logs: {result['tensorboard_logs']}")
        print(f"🔗 View training dashboard: https://modal.com/apps")
        print()
        print("🚀 TRAINING BREAKDOWN:")
        print("   💵 Total spent: ~$2-3")
        print("   ⏱️  Training time: ~15-20 minutes")
        print("   🔥 GPU: H100 (maximum performance)")
        print("   📊 Dataset: Filtered & optimized")
        print("   🔄 Resumption: Auto-resumes from last checkpoint")
        print("   🎯 Quality: Fast, efficient model!")
        print()
        print("💾 TO DOWNLOAD YOUR TRAINED MODELS:")
        print("   1. Go to: https://modal.com/storage")
        print("   2. Find volume: 'transformer-training-vol'")  
        print("   3. Download: weights/ folder (tam_model_*.pth checkpoints)")
        print("   4. Download: runs/ folder (TensorBoard logs with epochs)")
        print()
        print("🔄 CHECKPOINT INFO:")
        print("   📝 Naming: tam_model_1.pth, tam_model_2.pth, etc.")
        print("   🔄 Auto-resume: Run again to continue from last epoch")
        print("   📊 TensorBoard: Shows epoch numbers and progress")
        print()
        print("🚀 Your H100-trained transformer is ready!")
    else:
        print(f"\n❌ TRAINING FAILED: {result['message']}")
        print("💡 Check the logs above for details")

if __name__ == "__main__":
    main() 