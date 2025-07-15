import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from .ad_dataset import AudioClassificationDataset, AudioDatasetTransformer
from sklearn.model_selection import train_test_split
import os
import numpy as np

from transformers import AutoProcessor, AutoTokenizer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name)

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            if key == "features":
                max_len = max(item[key].size(0) for item in batch)
                padded_features = [
                    torch.nn.functional.pad(
                        item[key], (0, 0, 0, max_len - item[key].size(0))
                    )
                    for item in batch
                ]
                collated[key] = torch.stack(padded_features)
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([item[key] for item in batch])
        elif batch[0][key] is None:
            collated[key] = None
        else:
            collated[key] = [item[key] for item in batch]
    return collated

def plot_distributions(train_metadata, val_metadata, labels, save_path="train_val_distribution.png"):
    label_columns = labels.columns[1:]
    train_labels = labels[labels["uid"].isin(train_metadata["uid"])]
    val_labels = labels[labels["uid"].isin(val_metadata["uid"])]
    train_label_counts = train_labels[label_columns].sum(axis=0)
    val_label_counts = val_labels[label_columns].sum(axis=0)

    label_names = [str(col) for col in label_columns]
    x = range(len(label_names))
    plt.bar(x, train_label_counts, width=0.4, label="Train", align="center", color="blue")
    plt.bar([i + 0.4 for i in x], val_label_counts, width=0.4, label="Validation", align="center", color="orange")
    plt.xticks([i + 0.2 for i in x], label_names, rotation=45)
    plt.title("Label Distribution: Train vs Validation")
    plt.xlabel("Labels")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def get_dataloaders(config, processor):
    metadata_file = config["data"]["metadata_file"]
    labels_file = config["data"]["labels_file"]
    train_audio_dir = config["data"]["train_audio_dir"]
    test_audio_dir = config["data"]["test_audio_dir"]
    batch_size = config["training"]["batch_size"]
    augment = config["misc"]["augment"]
    selected_task = config["misc"]["selected_task"]

    metadata = pd.read_csv(metadata_file)
    labels = pd.read_csv(labels_file)


    labels["label_encoded"] = labels.iloc[:, 1:].idxmax(axis=1).map({
        "diagnosis_control": 0,
        "diagnosis_mci": 1,
        "diagnosis_adrd": 2,
        "diagnosis_other": 3,
    })

    train_metadata = metadata[metadata["split"] == "train"]
    test_metadata = metadata[metadata["split"] == "test"]
    train_metadata = train_metadata[train_metadata["uid"].isin(labels["uid"])]

    if config['model']['val_split']:
        train_uids, val_uids = train_test_split(
            train_metadata["uid"],
            test_size=0.1,
            random_state=42,
            stratify=labels.loc[labels["uid"].isin(train_metadata["uid"]), "label_encoded"]
        )

        train_metadata_post = train_metadata[train_metadata["uid"].isin(train_uids)]
        val_metadata_post = train_metadata[train_metadata["uid"].isin(val_uids)]
        # val_metadata_post.to_csv("./val.csv", index=False)

    train_dataset = AudioClassificationDataset(
        metadata=train_metadata_post if config['model']['val_split'] else train_metadata,
        labels=labels,
        audio_dir=train_audio_dir,
        processor=processor,
        augment=augment,
        include_metadata=config['model']['include_metadata'],
        random_segment=config['model']['random_segment'],
    )
    val_dataset = AudioClassificationDataset(
        metadata=val_metadata_post if config['model']['val_split'] else test_metadata,
        labels=labels,
        audio_dir=train_audio_dir if config['model']['val_split'] else test_audio_dir,
        processor=processor,
        augment=False,
        include_metadata=config['model']['include_metadata'],
        random_segment=config['model']['random_segment'],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader

def get_dataloaders_loo(config, processor):
    metadata_file = config["data"]["metadata_file"]
    loo_file = config["data"]["loo_file"]
    labels_file = config["data"]["labels_file"]
    audio_dir = config["data"]["audio_dir"]
    batch_size = config["training"]["batch_size"]
    augment = config["misc"]["augment"]
    leave_out = config["training"]["leave_out"]
    filter_col = "language" if leave_out in ["english","spanish","mandarin"] else "task"

    metadata = pd.read_csv(metadata_file)
    loo = pd.read_csv(loo_file)
    labels = pd.read_csv(labels_file)

    loo["unique_id"] = loo["unique_id"].astype(int)
    metadata['unique_id'] = metadata['uid'].apply(lambda x: int(str(x).split('_')[0]))

    metadata = pd.merge(metadata,loo[["unique_id",filter_col]],on="unique_id")
    metadata["audio_dir"] = metadata["split"]
    metadata["split"] = metadata[filter_col].apply(lambda x: "test" if x == leave_out else "train")

    labels["label_encoded"] = labels.iloc[:, 1:].idxmax(axis=1).map({
        "diagnosis_control": 0,
        "diagnosis_mci": 1,
        "diagnosis_adrd": 2,
        "diagnosis_other": 3,
    })

    train_metadata = metadata[metadata["split"] == "train"]
    test_metadata = metadata[metadata["split"] == "test"]
    train_metadata = train_metadata[train_metadata["uid"].isin(labels["uid"])]

    if config['model']['val_split']:
        train_uids, val_uids = train_test_split(
            train_metadata["uid"],
            test_size=0.2,
            random_state=42,
            stratify=labels.loc[labels["uid"].isin(train_metadata["uid"]), "label_encoded"]
        )

        train_metadata_post = train_metadata[train_metadata["uid"].isin(train_uids)]
        val_metadata_post = train_metadata[train_metadata["uid"].isin(val_uids)]

    train_dataset = AudioClassificationDataset(
        metadata=train_metadata_post if config['model']['val_split'] else train_metadata,
        labels=labels,
        audio_dir=audio_dir,
        processor=processor,
        augment=augment,
        include_metadata=config['model']['include_metadata'],
        random_segment=config['model']['random_segment'],
    )
    val_dataset = AudioClassificationDataset(
        metadata=val_metadata_post if config['model']['val_split'] else test_metadata,
        labels=labels,
        audio_dir=audio_dir,
        processor=processor,
        augment=False,
        include_metadata=config['model']['include_metadata'],
        random_segment=config['model']['random_segment'],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader

def _group_audio_features(df):
    df = df[df['segment_end_sec']<=300]
    return df

def get_dataloaders_transformer(config, processor):
    metadata_file = config["data"]["metadata_file"]
    labels_file = config["data"]["labels_file"]
    train_audio_dir = config["data"]["train_audio_dir"]
    test_audio_dir = config["data"]["test_audio_dir"]
    batch_size = config["training"]["batch_size"]
    augment = config["misc"]["augment"]
    include_features = config["model"]["include_features"]

    if include_features:
        train_features = _group_audio_features(pd.read_csv(config["data"]["train_audio_features"]))
        test_features = _group_audio_features(pd.read_csv(config["data"]["test_audio_features"]))


    metadata = pd.read_csv(metadata_file)
    labels = pd.read_csv(labels_file)

    cache_train = os.path.join(config['paths']['dataset_cache'],'train')
    cache_test = os.path.join(config['paths']['dataset_cache'],'test')
    os.makedirs(cache_train, exist_ok=True)
    os.makedirs(cache_test, exist_ok=True)


    labels["label_encoded"] = labels.iloc[:, 1:].idxmax(axis=1).map({
        "diagnosis_control": 0,
        "diagnosis_mci": 1,
        "diagnosis_adrd": 2,
        "diagnosis_other": 3,
    })

    train_metadata = metadata[metadata["split"] == "train"]
    test_metadata = metadata[metadata["split"] == "test"]
    train_metadata = train_metadata[train_metadata["uid"].isin(labels["uid"])]
    
    # Filter test by eureka subset
    if config["training"]["val_eureka"]:        
        eureka_val = pd.read_csv(config['data']['test_labels_eureka'])
        test_metadata = test_metadata[test_metadata['uid'].isin(eureka_val['uid'])]

    if config['model']['val_split']:
        train_uids, val_uids = train_test_split(
            train_metadata["uid"],
            test_size=0.1,
            random_state=42,
            stratify=labels.loc[labels["uid"].isin(train_metadata["uid"]), "label_encoded"]
        )

        train_metadata_post = train_metadata[train_metadata["uid"].isin(train_uids)]
        val_metadata_post = train_metadata[train_metadata["uid"].isin(val_uids)]
        # val_metadata_post.to_csv("./val.csv", index=False)

    train_dataset = AudioDatasetTransformer(
        metadata=train_metadata_post if config['model']['val_split'] else train_metadata,
        labels=labels,
        audio_dir=train_audio_dir,
        processor=processor,
        augment=augment,
        include_metadata=config['model']['include_metadata'],
        features=train_features if include_features else None,
        random_segment=config['model']['random_segment'],
        save_path=cache_train,
        max_segments=config['model']['max_segments'],
    )
    val_dataset = AudioDatasetTransformer(
        metadata=val_metadata_post if config['model']['val_split'] else test_metadata,
        labels=labels,
        audio_dir=train_audio_dir if config['model']['val_split'] else test_audio_dir,
        processor=processor,
        augment=False,
        include_metadata=config['model']['include_metadata'],
        features=test_features if include_features else None,
        random_segment=config['model']['random_segment'],
        save_path=cache_test,
        max_segments=config['model']['max_segments'],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader