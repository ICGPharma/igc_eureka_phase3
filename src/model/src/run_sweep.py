import os

import torch
import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pprint import pformat
import optuna
from utils.utils import load_config, get_dataloaders_transformer
from torch.utils.data import DataLoader
from utils.whisper_transformer import WhisperTransformerClassifier
from transformers import get_scheduler, AutoProcessor

## MLFLOW Setup
mlflow.set_tracking_uri("http://localhost:5000")
print('tracking uri:', mlflow.get_tracking_uri())

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device: torch.device,
        epoch,
        trial,
        scheduler,
    ):
    model.train()
    train_loss = 0.0
    y_true_train = []
    y_pred_train = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Trial {trial.number}", unit="batch")

    for batch in progress_bar:
        optimizer.zero_grad()

        labels = batch["label"].to(device)

        input_values = batch["input_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(audio_features=input_values,attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(preds)

        progress_bar.set_postfix(train_loss=loss.item())

    avg_loss = train_loss / len(train_loader)
    train_f1 = precision_recall_fscore_support(y_true_train, y_pred_train, average='weighted', zero_division=0)[2]
    torch.cuda.empty_cache()
    return avg_loss, train_f1


def validate_model(
        model: nn.Module,
        val_loader: DataLoader, 
        device: torch.device,
        criterion,
        trial
    ):
    model.eval()
    val_loss = 0    
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating - Trial {trial.number}", unit="batch"):
            labels = batch["label"].to(device)

            input_values = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(audio_features=input_values,attention_mask=attention_mask)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(preds)

            torch.cuda.empty_cache()
    avg_val_loss = val_loss / len(val_loader)
    test_accuracy = accuracy_score(y_true_val, y_pred_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted', zero_division=0)

    print(f'Accuracy of the model on the val data: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return avg_val_loss, f1

def suggest_hyperparameters(trial):
    num_heads_choices = [5,8,10,16,32]
    num_layers_choices = [3,6,10,12,14]
    activation_choices = ['relu','gelu']
    num_heads = trial.suggest_categorical("num_heads", num_heads_choices)
    num_layers = trial.suggest_categorical("num_layers", num_layers_choices)
    activation = trial.suggest_categorical("activation", activation_choices)
    lr = trial.suggest_float("lr", 1e-7, 1e-5, log=True)

    print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    return lr, num_heads, num_layers, activation
    
def objective(trial, experiment, device, config, processor):
    best_val_loss = float('inf')
    
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # mlflow.log_params(options)
        lr, num_heads, num_layers, activation = suggest_hyperparameters(trial)
        active_run = mlflow.active_run()
        print(f"Starting run {active_run.info.run_id} and trial {trial.number}")

        mlflow.log_param("lr", lr)
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("activation", activation)

        if num_layers==14:
            batch = 10
        elif num_layers==12:
            batch = 10
        elif num_layers==10:
            batch = 14
        elif num_layers==6:
            batch = 26
        elif num_layers==3:
            batch = 48
        config["training"]["batch_size"] = batch

        # Model parameters
        train_loader, val_loader = get_dataloaders_transformer(config, processor)

        model = WhisperTransformerClassifier(
            num_heads=num_heads,
            num_layers=num_layers,
            activation_encoder=activation,
        ).to(device)
        model = torch.nn.DataParallel(model)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
        train_steps = config["epochs"] * len(train_loader)
        warmup_steps = int(train_steps * config["warmup_ratio"])
        scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=train_steps,
            )
        criterion = nn.CrossEntropyLoss().to(device)

        for epoch in range(0, config["epochs"]):
            avg_train_loss, train_f1 = train_model(model, train_loader, optimizer, criterion, config["epochs"], device, epoch, trial, scheduler)
            avg_val_loss, val_f1 = validate_model(model, val_loader, device, criterion, trial)

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            trial.report(avg_val_loss, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
            
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f} - Train F1: {train_f1:.4f} - Val Loss: {avg_val_loss:.4f} - Val F1: {val_f1:.4f}")

    return best_val_loss

def main():
    config = load_config("../config/config_sweep.yaml")
    device = config['device']

    # Create mlflow experiment if it doesn't exist already
    experiment_name = config['experiment_name']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)

    model_id = config['training']['model_name']
    processor = AutoProcessor.from_pretrained(model_id)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    # Create the optuna study which shares the experiment name
    study = optuna.create_study(study_name=experiment_name, direction="minimize")
    study.optimize(
        lambda trial: objective(trial, experiment, device, config, processor),
        # n_trials=config['n_trials'],
        timeout=config['duration']*3600, # Convert hours to seconds
        show_progress_bar=True
    )

    # Filter optuna trials by state
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Value (Val Loss): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Log the best model
    if config["save_model"]:
        best_model_params = trial.params
        best_model = WhisperTransformerClassifier(
            num_heads=best_model_params['num_heads'],
            num_layers=best_model_params['num_layers'],
            activation_encoder=best_model_params['activation'],
        ).to(device)
        
        os.makedirs(config['checkpoints_path'], exist_ok=True)
        model_path = os.path.join(config['checkpoints_path'], f"best_model_trial_{trial.number}.pth")
        torch.save(best_model.state_dict(), model_path)
        print(f"Best model saved as {model_path}")

        mlflow.pytorch.log_model(best_model, "best_model", registered_model_name='best_transformer_model')
        mlflow.log_artifact(model_path, artifact_path="models")
        print(f"Best model saved as {model_path}")
        # Get the artifact URI dynamically

if __name__ == '__main__':
    main()