import os
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss, f1_score
import matplotlib.pyplot as plt
import json
import numpy as np
from transformers import get_scheduler
from torch.amp import autocast

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    device,
    optimizer,
    num_training_steps,
    num_warmup_steps,
    patience=5,
    checkpoints_path="../checkpoints/distilled_whisper",
    save_metrics_file="../results/distilled_whisper/metrics.json",
    save_epoch_data_file="../results/distilled_whisper/epoch_data.json",
    metrics_plot_file="../results/distilled_whisper/loss_f1_train_val.png",
    include_metadata=False,
    include_features=False,
    use_text=False,
    use_audio_text=False,
    num_labels=3,
    epoch_start=0,
):

    os.makedirs(checkpoints_path, exist_ok=True)  # Ensure the checkpoint directory exists

    # Create a warmup scheduler
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Weights = Total samples/(# classes * # samples)
    # weight_classes = torch.tensor([0.414, 0.931, 2.454, 9.270]).to(device)
    criterion = torch.nn.CrossEntropyLoss()  # PyTorch loss
    train_losses, val_losses, train_f1_scores, val_f1_scores = [], [], [], []
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_checkpoint = ""

    epoch_data = {"train": [], "validation": []}
    if epoch_start > 0:
        epochs += epoch_start

    for epoch in range(epoch_start,epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_logits = []
        train_labels = []

        # counter=0
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for batch in train_progress:
            optimizer.zero_grad()

            # Extract inputs based on model type
            labels = batch["label"].to(device)
            if len(labels.shape) > 1:
                labels = labels.squeeze(1)

            input_values = batch["input_features"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            # metadata=None
            # features=None
            # if include_metadata:
            #     metadata = torch.stack((batch['age'],batch['gender'],batch['education']),dim=1).to(device)
            # if include_features:
            #     features = batch['audio_features'].to(device)
                
            # logits = model(audio_features=input_values,attention_mask=attention_mask,metadata=metadata,features=features)#.logits
            logits = model(input_values).logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            train_loss += loss.item()

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_logits.append(probs)
            train_labels.append(y_true)
            loss_sklearn = log_loss(y_true, probs, labels=[x for x in range(num_labels)])
            preds = np.argmax(probs, axis=1)
            f1_train = f1_score(y_true, preds, average="macro")
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            train_progress.set_postfix({"loss_torch": loss.item(), "loss_sklearn": loss_sklearn, "f1_score": f1_train, "lr": current_lr})
            # if counter%10==0:
            #     print('Multiplier Value:',model.module.encoder.multiplier)
            # counter+=1
            # if counter==10:
            #     break
            torch.cuda.empty_cache()

        train_loss /= len(train_dataloader)
        train_logits = np.concatenate(train_logits, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        f1_train_epoch = f1_score(train_labels, np.argmax(train_logits, axis=1), average="macro")
        train_losses.append(train_loss)
        train_f1_scores.append(f1_train_epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_logits = []
        val_labels = []
        
        val_progress = tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            with autocast(device_type='cuda'):
                # counter=0
                for batch in val_progress:
                    labels = batch["label"].to(device)
                    if len(labels.shape) > 1:
                        labels = labels.squeeze(1)
                    
                    input_values = batch["input_features"].to(device)
                    # attention_mask = batch["attention_mask"].to(device)
                    # metadata=None
                    # features=None
                    # if include_metadata:
                    #     metadata = torch.stack((batch['age'],batch['gender'],batch['education']),dim=1).to(device)
                    # if include_features:
                    #     features = batch['audio_features'].to(device)
                    
                    # logits = model(audio_features=input_values,attention_mask=attention_mask,metadata=metadata,features=features)#.logits
                    logits = model(input_values).logits

                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    y_true = labels.detach().cpu().numpy()
                    val_logits.append(probs)
                    val_labels.append(y_true)
                    loss_sklearn = log_loss(y_true, probs, labels=[x for x in range(num_labels)])
                    preds = np.argmax(probs, axis=1)
                    f1_val = f1_score(y_true, preds, average="macro")
                    val_progress.set_postfix({"val_loss": loss.item(), "val_loss_sk": loss_sklearn, "val_f1_score": f1_val})
                    # counter+=1
                    # if counter==10:
                    #     break
                    torch.cuda.empty_cache()

        val_loss /= len(val_dataloader)
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        f1_val_epoch = f1_score(val_labels, np.argmax(val_logits, axis=1), average="macro")
        val_losses.append(val_loss)
        val_f1_scores.append(f1_val_epoch)

        epoch_data["train"].append({"logits": train_logits.tolist(), "labels": train_labels.tolist()})
        epoch_data["validation"].append({"logits": val_logits.tolist(), "labels": val_labels.tolist()})

        # Check if validation improved
        if val_loss < best_val_loss or (epoch+1)%10==0 or (epoch+1)==epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            checkpoint_path = os.path.join(checkpoints_path, f"checkpoint_epoch_{epoch + 1}.pth")
            best_checkpoint = checkpoint_path
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'f1_train_epoch': f1_train_epoch,
                    'f1_val_epoch': f1_val_epoch,
                }, checkpoint_path)
            if val_loss == best_val_loss:
                print(f"Validation loss improved! Checkpoint saved: {checkpoint_path}") 
            else:
                print(f"Checkpoint saved: {checkpoint_path}")
                early_stop_counter += 1
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            checkpoint_path = os.path.join(checkpoints_path, f"checkpoint_epoch_{epoch + 1}.pth")
            best_checkpoint = checkpoint_path
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'f1_train_epoch': f1_train_epoch,
                    'f1_val_epoch': f1_val_epoch,
                }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"Early stopping triggered after {patience} epochs of no improvement.")
            break

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train F1 Score: {f1_train_epoch:.4f} | Val F1 Score: {f1_val_epoch:.4f}")
        torch.cuda.empty_cache()

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_f1_scores": train_f1_scores,
        "val_f1_scores": val_f1_scores,
    }
    os.makedirs(os.path.dirname(save_metrics_file), exist_ok=True)
    os.makedirs(os.path.dirname(save_epoch_data_file), exist_ok=True)

    with open(save_metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(save_epoch_data_file, "w") as f:
        json.dump(epoch_data, f, indent=4)

    # Plotting (unchanged)
    plt.figure(figsize=(10, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # F1 score plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_f1_scores) + 1), train_f1_scores, label="Train F1 Score")
    plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, label="Val F1 Score")
    plt.title("F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(metrics_plot_file)
    plt.show()

    print("Training complete!")
    return train_losses, val_losses, train_f1_scores, val_f1_scores, best_checkpoint
