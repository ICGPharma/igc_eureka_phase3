# run_inference.py
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification, AutoProcessor, WhisperForAudioClassification, AutoConfig
from utils.custom_whisper import WhisperForAudioClassificationCustom, WhisperForAudioClassificationCustomEncoder
from utils.whisper_transformer import WhisperTransformerClassifier
from utils.ad_dataset import AudioClassificationDataset, AudioDatasetTransformer
from utils.utils import load_config, custom_collate_fn
from tqdm import tqdm
import torch.nn as nn

def main():
    # Load configuration
    config = load_config("../config/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths explicitly as in your submission_generate script
    metadata_file = config['data']['metadata_file']
    test_audio_dir = config['data']['test_audio_dir']

    # Load metadata for test set
    metadata = pd.read_csv(metadata_file)
    test_metadata = metadata[metadata["split"] == "test"]
    files = os.listdir('/buckets/projects/eureka/data_exp1/post_data/openvoice_translation_llama_V1')
    files = [int(x[:6]) for x in files if x!='tmp.wav']
    test_metadata = metadata[metadata['uid'].isin(files)].reset_index(drop=True)
    include_metadata = config['model']['include_metadata']
    include_features = config['model']['include_features']

    # Model setup
    model_id = config['training']['model_name']

    if config['training']['custom']:
        config_model = AutoConfig.from_pretrained(model_id, num_labels=config["training"]["num_labels"])
        # model = WhisperForAudioClassificationCustom(config_model)
        # model = WhisperForAudioClassificationCustomEncoder(config_model)
        model = WhisperTransformerClassifier(
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            include_metadata=include_metadata,
            include_features=include_features,
            max_segments=config['model']['max_segments'],
            mlp_classifier=config['model']['mlp_classifier'],
        )
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            num_labels=config["training"]["num_labels"], 
        )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config["paths"]["final_model"]
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # # Create test dataset and DataLoader
    # test_dataset = AudioClassificationDataset(
    #     metadata=test_metadata,
    #     labels=None,
    #     audio_dir=test_audio_dir,
    #     processor=processor,
    #     augment=False,
    #     test=True,
    #     include_metadata=include_metadata,
    # )
    test_features = None
    if include_features:
        test_features = pd.read_csv(config["data"]["test_audio_features"])
        test_features = test_features[test_features['segment_end_sec']<=300]

    test_dataset = AudioDatasetTransformer(
        metadata=test_metadata,
        labels=None,
        audio_dir=test_audio_dir,
        # audio_dir='/buckets/projects/eureka/data_exp1/post_data/openvoice_translation_llama_V1',
        processor=processor,
        augment=False,
        test=True,
        include_metadata=include_metadata,
        features=test_features,
        save_path=os.path.join(config['paths']['dataset_cache'],'test'),
        max_segments=config['model']['max_segments'],
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        collate_fn=custom_collate_fn
    )

    # Inference and submission file generation logic
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Submission"):
            input_values = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata=None
            features=None
            if include_metadata:
                metadata = torch.stack((batch['age'],batch['gender'],batch['education']),dim=1).to(device)
            if include_features:
                features = batch['audio_features'].to(device)
                
            logits = model(
                audio_features=input_values,
                attention_mask=attention_mask,
                metadata=metadata,
                features=features
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for uid, probs in zip(batch["uid"], probabilities):
                if type(uid)==torch.Tensor:
                    uid = uid.item()
                predictions.append([uid] + probs.tolist())

    # Save predictions
    submission_df = pd.DataFrame(
        predictions, 
        columns=["uid", "diagnosis_control", "diagnosis_mci", "diagnosis_adrd", "diagnosis_other"]
    )
    submission_file = os.path.join(config["paths"]["results_path"], "submission_whisper_final.csv")
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    submission_df.to_csv(submission_file, index=False)

    print(f"Submission saved to {submission_file}")

if __name__ == "__main__":
    main()