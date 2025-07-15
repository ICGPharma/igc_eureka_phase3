import torch
import torch.nn as nn
from transformers import WhisperForAudioClassification, AutoConfig

class WhisperTransformerClassifier(nn.Module):
    def __init__(
            self,
            num_classes=4,
            num_layers=6,
            num_heads=8,
            include_metadata=False,
            include_features=False,
            activation_encoder='relu',
            max_segments=16,
            mlp_classifier=False,
        ):
        super().__init__()
        
        self.hidden_size = 1280 # From Whisper
        self.max_segments = max_segments
        self.segment_embedding = nn.Embedding(self.max_segments, self.hidden_size)

        self.include_metadata = include_metadata
        self.include_features = include_features

        if self.include_metadata:
            self.num_age_bins = 54 # max:99, min:46
            self.embedding_age = nn.Embedding(self.num_age_bins+1, self.hidden_size, padding_idx=0)
            self.num_gender_bins = 2
            self.embedding_gender = nn.Embedding(self.num_gender_bins+1, self.hidden_size, padding_idx=0)
            self.num_edu_bins = 23 # max:22, min:0
            self.embedding_edu = nn.Embedding(self.num_edu_bins+1, self.hidden_size, padding_idx=0)
        
        if self.include_features:
            layers=[]
            layers.append(nn.Linear(27000, 13500))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(13500, 6750))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(6750, 3375))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(3375, 1680))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(1680, 840))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(840, 1280))
            layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        # input_dim = 1280
        # feedforward dim = 2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            batch_first=True,
            activation=activation_encoder,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if mlp_classifier:
            layers = []
            layers.append(nn.Linear(self.hidden_size, self.hidden_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_size//2, self.hidden_size//4))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_size//4, self.hidden_size//8))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_size//8, num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, audio_features, attention_mask, metadata=None, features=None):
        B, _, _ = audio_features.shape #[Batch, Segments, Hidden Size from Whisper]
        device = audio_features.device
        position = torch.repeat_interleave(torch.arange(0, self.max_segments), 1500).unsqueeze(0).repeat(B, 1).to(device)

        segment_emb = self.segment_embedding(position)
        x = audio_features + segment_emb

        if self.include_metadata:
            age = metadata[:,0] + 1 # + 1 for deplacement of pad_idx
            age = age - 46 # 46 is the min age
            age = torch.where(age < 0, torch.tensor(0, dtype=torch.long), age)
            embed_age = self.embedding_age(age.int()) # (B,1280)
            embed_age = embed_age.unsqueeze(1)

            gender = metadata[:,1] + 1 # + 1 for deplacement of pad_idx
            embed_gender = self.embedding_gender(gender.int()) # (B,1280)
            embed_gender = embed_gender.unsqueeze(1)

            edu = metadata[:,2] + 1 # + 1 for deplacement of pad_idx
            embed_edu = self.embedding_edu(edu.int()) # (B,1280)
            embed_edu = embed_edu.unsqueeze(1)

            values = torch.stack([age, gender, edu], dim=0)
            values_T = values.T
            metadata_mask = (values_T == 0).int()

        cls_token = self.cls_token.expand(B, -1, -1)

        if self.include_metadata:
            x = torch.cat([cls_token, embed_age, embed_gender, embed_edu, x], dim=1)
        else:
            x = torch.cat([cls_token, x], dim=1)

        if self.include_metadata:
            attention_mask = torch.cat([torch.tensor(B*[[0.0]]).to(device),metadata_mask,attention_mask],dim=1).bool()
        else:
            attention_mask = torch.cat([torch.tensor(B*[[0.0]]).to(device),attention_mask],dim=1).bool()

        out = self.transformer(x, src_key_padding_mask=attention_mask)
        cls_out = out[:, 0, :]  # [B, H]

        if self.include_features:
            mlp_out = self.mlp(features)
            cls_out = cls_out + mlp_out

        return self.classifier(cls_out)

