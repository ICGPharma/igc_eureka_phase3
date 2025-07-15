# run_train.py
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from utils.utils import load_config, get_dataloaders, get_dataloaders_loo, custom_collate_fn
from utils.train import train_model
from transformers import AutoModelForAudioClassification, AutoProcessor, WhisperForAudioClassification, AutoConfig
from utils.custom_whisper import WhisperForAudioClassificationCustom, WhisperForAudioClassificationCustomEncoder
import torch.nn as nn
from utils.ad_dataset import AudioClassificationDataset

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
    save_metrics_file = os.path.join(results_path, "metrics.json")
    save_epoch_data_file = os.path.join(results_path, "epoch_data.json")
    metrics_plot_file = os.path.join(results_path, "loss_f1_train_val.png")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    model_id = config['training']['model_name']

    if config['training']['custom']:
        config_model = AutoConfig.from_pretrained(model_id, num_labels=config["training"]["num_labels"])
        # model = WhisperForAudioClassificationCustom(config_model)
        # model.load_state_dict(
        #     WhisperForAudioClassification.from_pretrained(model_id, config=config_model).state_dict(),
        #     strict=False
        # )
        model = WhisperForAudioClassificationCustomEncoder(config_model)
        if config["training"]["load_checkpoint"]:
            print('Load from base checkpoint')
            checkpoint_path = config["paths"]["base_checkpoint"]
            state_dict = torch.load(checkpoint_path, weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "", 1)  # Only replace the first occurrence
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
            del state_dict
        else:
            model.load_state_dict(
                WhisperForAudioClassification.from_pretrained(model_id, config=config_model).state_dict(),
                strict=False
            )
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            num_labels=config["training"]["num_labels"], 
        )

    processor = AutoProcessor.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if config["training"]["freeze"]:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze Conv layers, Embedding, Layer norm after encoder, and projector & classifier 
        for param in model.encoder.conv1.parameters():
            param.requires_grad = True
        for param in model.encoder.conv2.parameters():
            param.requires_grad = True
        # model.encoder.multiplier.requires_grad = True # TODO: Change when using custom encoder
        for param in model.encoder.embed_positions.parameters():
            param.requires_grad = True
        for param in model.encoder.layer_norm.parameters():
            param.requires_grad = True
        for param in model.projector.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        num_layers = len(model.encoder.layers)
        keep_first_n = 2
        keep_last_m = 1
        for i in list(range(keep_first_n)) + list(range(num_layers - keep_last_m, num_layers)):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # if config["training"]["load_checkpoint"]:
    #     print('Running inference to validate checkpoint')
    #     metadata_file = config['data']['metadata_file']
    #     test_audio_dir = config['data']['test_audio_dir']
    #     metadata = pd.read_csv(metadata_file)
    #     test_metadata = metadata[metadata["split"] == "test"]
    #     include_metadata = config['model']['include_metadata']
    #     test_dataset = AudioClassificationDataset(
    #         metadata=test_metadata,
    #         labels=None,
    #         audio_dir=test_audio_dir,
    #         processor=processor,
    #         augment=False,
    #         test=True,
    #         include_metadata=include_metadata,
    #     )
    #     test_loader = DataLoader(
    #         test_dataset, 
    #         batch_size=50,  # Adjust batch size if necessary
    #         shuffle=False, 
    #         collate_fn=custom_collate_fn
    #     )
    #     # Inference and submission file generation logic
    #     predictions = []

    #     with torch.no_grad():
    #         for batch in tqdm(test_loader, desc="Generating Submission"):
    #             input_values = batch["input_features"].to(device)
    #             if include_metadata:
    #                 # metadata = torch.stack((batch['age'],batch['gender'],batch['education']),dim=1).to(device)
    #                 metadata = batch['age'].to(device)
    #                 logits = model(input_values,metadata_features=metadata).logits
    #             else:
    #                 logits = model(input_values).logits
    #             probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    #             for uid, probs in zip(batch["uid"], probabilities):
    #                 predictions.append([uid] + probs.tolist())
    #     submission_df = pd.DataFrame(
    #         predictions, 
    #         columns=["uid", "diagnosis_control", "diagnosis_mci", "diagnosis_adrd", "diagnosis_other"]
    #     )
    #     submission_file = os.path.join(config["paths"]["results_path"], "validate_loaded_model.csv")
    #     os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    #     submission_df.to_csv(submission_file, index=False)
    #     print(f"Inference validation saved to {submission_file}")
    #     del test_loader, test_dataset

    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    # Data loaders
    if config["training"]["leave_out"]!="":
        print("Leave out:",config["training"]["leave_out"])
        train_loader, val_loader = get_dataloaders_loo(config, processor)
    else:
        train_loader, val_loader = get_dataloaders(config, processor)

    # Training Stage 1: 4 epochs with validation
    epochs_stage1 = config["training"]["epochs"]
    # epochs_stage1 = 4
    train_steps_stage1 = epochs_stage1 * len(train_loader)
    warmup_steps_stage1 = int(train_steps_stage1 * config["training"]["warmup_ratio"])
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # print("Starting Training Stage 1: 4 epochs with validation")
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
        num_labels=config["training"]["num_labels"],
    )

    # # Combine train and validation datasets for Stage 2
    # combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    # combined_loader = DataLoader(combined_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # # Training Stage 2: additional 2 epochs
    # epochs_stage2 = 2
    # train_steps_stage2 = epochs_stage2 * len(combined_loader)
    # warmup_steps_stage2 = int(train_steps_stage2 * config["training"]["warmup_ratio"])

    # save_metrics_file = os.path.join(results_path, "metrics_2.json")
    # save_epoch_data_file = os.path.join(results_path, "epoch_data_2.json")
    # metrics_plot_file = os.path.join(results_path, "loss_f1_train_val_2.png")

    # print("Starting Training Stage 2: additional 2 epochs on full dataset")
    # train_model(
    #     model=model,
    #     train_dataloader=combined_loader,
    #     val_dataloader=val_loader,
    #     epochs=epochs_stage2,
    #     device=device,
    #     optimizer=optimizer,
    #     num_training_steps=train_steps_stage2,
    #     num_warmup_steps=warmup_steps_stage2,
    #     patience=config["training"]["patience"],
    #     checkpoints_path=checkpoints_path,
    #     save_metrics_file=save_metrics_file,
    #     save_epoch_data_file=save_epoch_data_file,
    #     metrics_plot_file=metrics_plot_file,
    #     epoch_start=epochs_stage1,
    #     num_labels=config["training"]["num_labels"],
    # )

    # Save final model
    final_model_path = os.path.join(checkpoints_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()