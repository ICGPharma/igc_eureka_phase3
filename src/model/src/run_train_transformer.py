# run_train.py
import torch
import os
from torch.optim import AdamW
from utils.utils import load_config, get_dataloaders_transformer
from utils.train import train_model
from transformers import AutoProcessor, AutoConfig
from utils.whisper_transformer import WhisperTransformerClassifier

def main():
    # Load configuration
    config = load_config("../config/config.yaml")
    # Set visible GPUs if specified
    gpus = config["training"].get("gpus", None)
    if gpus and gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus.strip()

    device = torch.device(config["training"]["device"]) if torch.cuda.is_available() else torch.device("cpu")

    results_path = config["paths"]["results_path"]
    checkpoints_path = config["paths"]["checkpoints_path"]
    cache_path = config["paths"]["dataset_cache"]
    save_metrics_file = os.path.join(results_path, "metrics.json")
    save_epoch_data_file = os.path.join(results_path, "epoch_data.json")
    metrics_plot_file = os.path.join(results_path, "loss_f1_train_val.png")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(cache_path, exist_ok=True)

    model_id = config['training']['model_name']
    processor = AutoProcessor.from_pretrained(model_id)

    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    # Data loaders
    train_loader, val_loader = get_dataloaders_transformer(config, processor)

    model = WhisperTransformerClassifier(
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        include_metadata=config['model']['include_metadata'],
        include_features=config['model']['include_features'],
        max_segments=config['model']['max_segments'],
        mlp_classifier=config['model']['mlp_classifier'],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    epochs_stage1 = config["training"]["epochs"]
    train_steps_stage1 = epochs_stage1 * len(train_loader)
    warmup_steps_stage1 = int(train_steps_stage1 * config["training"]["warmup_ratio"])
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Starting Training")
    train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs_stage1,
        device=device,
        optimizer=optimizer,
        num_training_steps=train_steps_stage1,
        num_warmup_steps=warmup_steps_stage1,
        patience=config["training"]["patience"],
        checkpoints_path=checkpoints_path,
        save_metrics_file=save_metrics_file,
        save_epoch_data_file=save_epoch_data_file,
        metrics_plot_file=metrics_plot_file,
        include_metadata=config["model"]["include_metadata"],
        include_features=config["model"]["include_features"],
        num_labels=config["training"]["num_labels"],
    )

    # Save final model
    final_model_path = os.path.join(checkpoints_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()