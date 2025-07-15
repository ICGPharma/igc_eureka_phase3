import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch_audiomentations import Compose, AddColoredNoise, PitchShift, Gain
from transformers import Wav2Vec2Processor, WhisperForAudioClassification, AutoConfig
from transformers import AutoTokenizer
import math
import pickle
from tqdm import tqdm
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import traceback
 
# Define augmentations
augmentations = Compose(
    transforms=[
        AddColoredNoise(min_snr_in_db=0, max_snr_in_db=18, p=0.5, sample_rate=16000, output_type="tensor"),
        PitchShift(sample_rate=16000, min_transpose_semitones=-5, max_transpose_semitones=5, p=0.5, output_type="tensor"),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5, output_type="tensor"),
    ],
    output_type="tensor",
)
 
class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        metadata,
        labels,
        audio_dir,
        processor,
        augment=False,
        target_sr=16000,
        max_length=30,
        test=False,
        include_metadata=False,
        random_segment=False,
        ):
        """
        Dataset for audio classification.
 
        Args:
            metadata (pd.DataFrame): Metadata containing information like uid, age, and gender.
            labels (pd.DataFrame or None): DataFrame containing labels. None for the test dataset.
            audio_dir (str): Path to the directory containing audio files.
            processor (Wav2Vec2Processor): Processor for audio feature extraction.
            augment (bool): Whether to apply augmentations.
            target_sr (int): Target sampling rate for audio files.
            max_length (float): Maximum length of audio in seconds.
            test (bool): Whether this dataset is for the test set (no labels).
        """
        self.metadata = metadata
        self.labels = labels  # Keep labels DataFrame if available
        self.audio_dir = audio_dir
        self.processor = processor
        self.augment = augment
        self.target_sr = target_sr
        self.max_length_samples = int(target_sr * max_length)
        self.test = test
        self.include_metadata = include_metadata
        self.random_segment = random_segment
 
    def __len__(self):
        return len(self.metadata)
 
    def __getitem__(self, idx):
        # Get metadata row
        row = self.metadata.iloc[idx]
        uid = row["uid"]
        if "audio_dir" in self.metadata.keys():
            audio_dir = os.path.join(self.audio_dir,row["audio_dir"]+"_audios")
        else:
            audio_dir = self.audio_dir
        audio_path = os.path.join(audio_dir, f"{uid}.mp3")
        # audio_path = os.path.join(self.audio_dir, f"{uid}_output_v2_en-newest.wav") # TODO: Change path when evaluating on translations
 
        # Retrieve label only if not a test dataset
        label = None
        if not self.test and self.labels is not None:
            label_row = self.labels[self.labels["uid"] == uid]
            if not label_row.empty:
                label = label_row["label_encoded"].values[0]  # Get encoded label
 
        if self.include_metadata:
            # Extract age and gender from metadata
            age = row["age"]
            gender = 0 if row["gender"] == "Male" else 1  # Encode gender
            education = row["education"]
 
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] == 2:
            # Convert to mono by averaging the two channels
            waveform = waveform.mean(dim=0, keepdim=True)
 
        # Resample if necessary
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
        # Truncate or pad waveform to max_length_samples
        if waveform.shape[1] > self.max_length_samples:
            if self.random_segment:
                max_start = waveform.shape[1] - self.max_length_samples
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start:start + self.max_length_samples]
            else:
                waveform = waveform[:, :self.max_length_samples]
        else:
            padding = self.max_length_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
 
        # Apply augmentations if enabled
        if self.augment:
            waveform = augmentations(waveform.unsqueeze(0), sample_rate=self.target_sr).squeeze(0)
 
        # Process audio with the Wav2Vec2 processor
        inputs = self.processor(
            waveform.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt", padding=True
        )
 
        # Return the sample as a dictionary
        return {
            "uid": uid,
            "input_features": inputs["input_features"].squeeze(0),
            # "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long) if label is not None else None,
            **({
                "age": torch.tensor(age, dtype=torch.float),
            } if self.include_metadata else {})
            # **({
            #     "age": torch.tensor(age, dtype=torch.float),
            #     "gender": torch.tensor(gender, dtype=torch.float),
            #     "education": torch.tensor(education, dtype=torch.float),
            # } if self.include_metadata else {})
        }

class AudioDatasetTransformer(Dataset):
    def __init__(
        self,
        metadata,
        labels,
        audio_dir,
        processor,
        save_path,
        augment=False,
        target_sr=16000,
        max_length=30,
        test=False,
        include_metadata=False,
        features=None,
        random_segment=False,
        max_segments=16,
        ):
        """
        Dataset for audio classification.
 
        Args:
            metadata (pd.DataFrame): Metadata containing information like uid, age, and gender.
            labels (pd.DataFrame or None): DataFrame containing labels. None for the test dataset.
            audio_dir (str): Path to the directory containing audio files.
            processor (Wav2Vec2Processor): Processor for audio feature extraction.
            augment (bool): Whether to apply augmentations.
            target_sr (int): Target sampling rate for audio files.
            max_length (float): Maximum length of audio in seconds.
            test (bool): Whether this dataset is for the test set (no labels).
        """
        self.metadata = metadata
        self.labels = labels  # Keep labels DataFrame if available
        self.audio_dir = audio_dir
        self.processor = processor
        self.augment = augment
        self.target_sr = target_sr
        self.max_length_samples = int(target_sr * max_length)
        self.test = test
        self.include_metadata = include_metadata
        self.features = features
        self.random_segment = random_segment
        self.max_segments = max_segments
        self.num_gpus = torch.cuda.device_count()

        self.data_paths = []
        self.save_path = save_path
        if os.path.exists(os.path.join(self.save_path,'all_paths.pkl')):
            self._load_data_from_cache()
            if len(self.data_paths)!=self.metadata.shape[0]:
                data_ids = list(self.metadata['uid'])
                self.data_paths = [x for x in self.data_paths if int(x.split('/')[-1].split('.')[0]) in data_ids]
        else:
            config_model = AutoConfig.from_pretrained('openai/whisper-large-v3', num_labels=4)
            self.whisper_models = [WhisperForAudioClassification(config_model) for _ in range(self.num_gpus)]
            state_dict = torch.load("../checkpoints/whisper-large_fullaudio/final_model.pth", weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "", 1)
                new_state_dict[new_key] = v
            for i,whisper in enumerate(self.whisper_models):
                whisper.load_state_dict(new_state_dict, strict=False)
                whisper.to(f"cuda:{i}")

            # self.whisper = torch.nn.DataParallel(self.whisper)
            
            # self.fit_transform()
            self.fit_transform_segments_parallel()
            # self.fit_transform_segments_single_thread()

            for whisper in self.whisper_models:
                whisper.to("cpu")  
            # self.whisper.module.to("cpu")
            torch.cuda.empty_cache()
    
    def save(self):
        with open(os.path.join(self.save_path,'all_paths.pkl'), 'wb') as file:
            pickle.dump(self.data_paths, file)
        print(f'Data paths saved to {os.path.join(self.save_path,'all_paths.pkl')}')

    def _load_data_from_cache(self):
        with open(os.path.join(self.save_path,'all_paths.pkl'), 'rb') as f:
            data_list = pickle.load(f)
            self.data_paths = data_list
        print(f"Data paths loaded from {os.path.join(self.save_path,'all_paths.pkl')}")

    def __len__(self):
        return len(self.metadata)
 
    def __getitem__(self, idx):
        path = self.data_paths[idx]
        with open(path, 'rb') as f:
            item_dict = pickle.load(f)
            # Trim segments if needed
            item_dict['input_features'] = item_dict['input_features'][:self.max_segments*1500]
            item_dict['attention_mask'] = item_dict['attention_mask'][:self.max_segments*1500]
        if self.include_metadata:
            row = self.metadata[self.metadata['uid']==item_dict['uid']].iloc[0]
            item_dict['age'] = row['age']
            item_dict['gender'] = -1 if row["gender"]=="-1" else (0 if row["gender"]=="Male" else 1)
            item_dict['education'] = row['education']
        # if self.features!=None:
        #     row_mask = self.features[0]==item_dict['uid']
        #     item_dict['audio_features'] = self.features[1][row_mask].view(-1) # Flatten
        #     item_dict['audio_features'] = torch.nn.functional.pad(
        #         item_dict['audio_features'],
        #         (0, 27000 - item_dict['audio_features'].size(0))
        #     )
        if self.features is not None:
            uid = item_dict['uid']
            user_df = self.features[self.features['uid'] == uid]
            user_df = user_df.drop(columns=['uid'])

            # Compute average chunks of 10
            chunks = []
            values = user_df.values
            for i in range(0, len(values), 10):
                chunk = values[i:i + 10]
                chunks.append(chunk.mean(axis=0))

            features_tensor = torch.tensor(np.vstack(chunks), dtype=torch.float32).view(-1)

            # Pad to fixed length (27000)
            pad_len = 27000 - features_tensor.shape[0]
            if pad_len > 0:
                features_tensor = F.pad(features_tensor, (0, pad_len))

            item_dict['audio_features'] = features_tensor
        return item_dict
    
    def _process_single_uid_no_overlap(self, row, model_index, lock):
        base_attention_mask = torch.zeros(self.max_segments*1500, dtype=torch.long)
        uid = row["uid"]
        output_path = os.path.join(self.save_path,f'{uid}.pkl')
        if os.path.exists(output_path):
            with lock:
                self.data_paths.append(output_path)
            return
        # audio_path = os.path.join(self.audio_dir, f"{uid}_output_v2_en-newest.wav") # TODO: Change to include route to translated audios
        audio_path = os.path.join(self.audio_dir, f"{uid}.mp3") # TODO: Change to include route to translated audios
    
        # Retrieve label only if not a test dataset
        label = None
        if not self.test and self.labels is not None:
            label_row = self.labels[self.labels["uid"] == uid]
            if not label_row.empty:
                label = label_row["label_encoded"].values[0]  # Get encoded label
    
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
        # Truncate or pad waveform to max_length_samples
        if waveform.shape[1] > self.max_length_samples*self.max_segments:
            waveform = waveform[:, :self.max_length_samples*self.max_segments]
            
        segments = []
        classifications = []
        for i in range(math.ceil(waveform.shape[1]/self.max_length_samples)):
            init_sec = self.max_length_samples * i
            end_sec = self.max_length_samples * (i + 1)
            end_sec = end_sec if end_sec < waveform.shape[1] else waveform.shape[1]

            segment = waveform[:, init_sec:end_sec]
            if segment.shape[1] < self.max_length_samples:
                padding = self.max_length_samples - end_sec + init_sec
                segment = torch.nn.functional.pad(segment, (0, padding))
            if self.augment:
                with self.processor_lock:
                    segment = augmentations(segment.unsqueeze(0), sample_rate=self.target_sr).squeeze(0)
    
            # Process audio with the Wav2Vec2 processor
            segment = self.processor(
                segment.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt", padding=True
            )
                
            input_features = segment["input_features"].to(f"cuda:{model_index}")
            whisper = self.whisper_models[model_index]
            with self.model_locks[model_index]:
                encoder_out = whisper.encoder(input_features=input_features).last_hidden_state
                encoder_out = encoder_out.detach().cpu().squeeze(0)

                classifier_out = whisper(input_features).logits
                classifier_out = classifier_out.detach().cpu().squeeze(0)

            segments.append(encoder_out)
            classifications.append(classifier_out)

        # Return the sample as a dictionary
        audio_features = torch.stack(segments).reshape(-1,1280)
        classifier = torch.stack(classifications)
        classifier = classifier.mean(dim=0)
        probs = torch.softmax(classifier,dim=0)
            
        attention_mask = base_attention_mask.clone()
        attention_mask[audio_features.shape[0]:] = 1

        pad_rows = (self.max_segments*1500) - audio_features.shape[0]
        padding = (0, 0, 0, pad_rows)
        audio_features = F.pad(audio_features, padding, value=0)
            
        with open(os.path.join(self.save_path,f'{uid}.pkl'), 'wb') as file:
            pickle.dump({
                    "uid": uid,
                    "input_features": audio_features,
                    "attention_mask": attention_mask,
                    "label": torch.tensor(label, dtype=torch.long) if label is not None else None,
                    "probs": probs,
                }, file)
        with lock:
            self.data_paths.append(output_path)
    
    def _process_single_uid(self, row, model_index, lock):
        uid = row["uid"]
        output_path = os.path.join(self.save_path, f'{uid}.pkl')
        if os.path.exists(output_path):
            with lock:
                self.data_paths.append(output_path)
            return
        try:
            audio_path = os.path.join(self.audio_dir, f"{uid}.mp3")
            label = None
            if not self.test and self.labels is not None:
                label_row = self.labels[self.labels["uid"] == uid]
                if not label_row.empty:
                    label = label_row["label_encoded"].values[0]

            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)

            total_seconds = waveform.shape[1] // self.target_sr
            trimmed_samples = total_seconds * self.target_sr
            max_total_samples = 300 * self.target_sr  # 5 minutes
            # waveform = waveform[:, :min(trimmed_samples,max_total_samples)]
            waveform = waveform[:, :max_total_samples]

            chunk_duration = 10  # seconds
            stride = 5           # seconds
            samples_per_chunk = chunk_duration * self.target_sr
            stride_samples = stride * self.target_sr
            required_samples = 30 * self.target_sr

            segments = []
            num_chunks = math.ceil((waveform.shape[1] - samples_per_chunk) / stride_samples)
            num_chunks = num_chunks + 1 if num_chunks > 0 else 1
            for i in range(num_chunks):
                start = i * stride_samples
                end = start + samples_per_chunk
                if start >= waveform.shape[1]:
                    break
                segment = waveform[:, start:end]
                if segment.shape[1] < samples_per_chunk:
                    segment = F.pad(segment, (0, samples_per_chunk - segment.shape[1]), value=0.0)
                if self.augment:
                    with self.processor_lock:
                        segment = augmentations(segment.unsqueeze(0), sample_rate=self.target_sr).squeeze(0)

                segment_len = segment.shape[1]
                if segment_len < required_samples:
                    pad_len = required_samples - segment_len
                    segment = F.pad(segment, (0, pad_len), value=0.0)
                elif segment_len > required_samples:
                    segment = segment[:, :required_samples]
                segment = self.processor(segment.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt", padding=False)
                segment_wave = segment["input_features"]  # [1, seq_len, 80]

                segment_wave = segment_wave.to(f"cuda:{model_index}")
                whisper = self.whisper_models[model_index]
                with self.model_locks[model_index]:
                    with torch.no_grad():
                        encoder_out = whisper.encoder(input_features=segment_wave).last_hidden_state  # [1, 1500, 1280]
                        encoder_out = encoder_out.detach().cpu().squeeze(0)
                segments.append(encoder_out)

            audio_features = torch.cat(segments, dim=0)  # [n*1500, 1280]
            max_total_frames = self.max_segments * 1500
            actual_frames = audio_features.shape[0]
            if actual_frames < max_total_frames:
                audio_features = F.pad(audio_features, (0, 0, 0, max_total_frames - actual_frames), value=0.0)
            else:
                audio_features = audio_features[:max_total_frames]

            attention_mask = torch.zeros(max_total_frames, dtype=torch.long)
            attention_mask[actual_frames:] = 1

            with open(output_path, 'wb') as f:
                pickle.dump({
                    "uid": uid,
                    "input_features": audio_features,
                    "attention_mask": attention_mask,
                    "label": torch.tensor(label, dtype=torch.long) if label is not None else None
                }, f)

            with lock:
                self.data_paths.append(output_path)
        except Exception as e:
            raise Exception(f"❌ Error processing UID={uid}: {traceback.print_exc()}")
    
    def fit_transform_segments_single_thread(self):
        lock = threading.Lock()
        self.model_locks = [threading.Lock() for _ in self.whisper_models]
        self.processor_lock = threading.Lock()
        for _,row in tqdm(self.metadata.iterrows(),total=len(self.metadata)):
            self._process_single_uid_no_overlap(row,0,lock)
    
    def fit_transform_segments_parallel(self):
        lock = threading.Lock()
        self.model_locks = [threading.Lock() for _ in self.whisper_models]
        self.processor_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            for i, (_, row) in enumerate(self.metadata.iterrows()):
                model_index = i % len(self.whisper_models)
                # futures.append(executor.submit(self._process_single_uid, row, model_index, lock))
                futures.append(executor.submit(self._process_single_uid_no_overlap, row, model_index, lock))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio"):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Error during processing: {e}")
        
        self.save()
 
class AudioClassificationDatasetF(Dataset):
    def __init__(self, metadata, labels, audio_dir, features_file, processor, augment=False, target_sr=16000, max_length=30, test=False):
        self.metadata = metadata
        self.labels = labels
        self.audio_dir = audio_dir
        self.features = pd.read_csv(features_file)
        self.processor = processor
        self.augment = augment
        self.target_sr = target_sr
        self.max_length_samples = int(target_sr * max_length)
        self.test = test
 
    def __len__(self):
        return len(self.metadata)
 
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["uid"]
        audio_path = os.path.join(self.audio_dir, f"{uid}.mp3")
 
        label = None
        if not self.test and self.labels is not None:
            label_row = self.labels[self.labels["uid"] == uid]
            if not label_row.empty:
                label = label_row["label_encoded"].values[0]
 
        age = row["age"]
        gender = 0 if row["gender"] == "male" else 1
 
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
 
        if waveform.shape[1] > self.max_length_samples:
            waveform = waveform[:, :self.max_length_samples]
        else:
            padding = self.max_length_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
 
        if self.augment:
            waveform = augmentations(waveform.unsqueeze(0), sample_rate=self.target_sr).squeeze(0)
 
        inputs = self.processor(waveform.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt", padding=True)
 
        feature_rows = self.features[self.features["uid"] == uid]
        features = torch.tensor(feature_rows.iloc[:, 3:].values, dtype=torch.float32)
 
        return {
            "uid": uid,
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long) if label is not None else None,
            "age": torch.tensor(age, dtype=torch.float),
            "gender": torch.tensor(gender, dtype=torch.float),
            "features": features,
        }
    
 
# Initialize BERT tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
 
class AudioTextClassificationDataset(Dataset):
    def __init__(self, metadata, labels, audio_dir, features_file, text_file, processor, augment=False, target_sr=16000, max_length=30, test=False, task=None):
        self.metadata = metadata
        self.labels = labels
        self.audio_dir = audio_dir
        self.features = pd.read_csv(features_file)
        self.text_data = pd.read_csv(text_file)
        self.processor = processor
        self.augment = augment
        self.target_sr = target_sr
        self.max_length_samples = int(target_sr * max_length)
        self.test = test
        self.task = task
        self.task_map = {'image':0, 'alexa':1, 'animals':2, 'quijote':3}
        self.lang_map = {'bg':6, 'en':0, 'es':1, 'gl':3, 'la':4, 'yo':5, 'zh':2, 'ru':7}
 
    def __len__(self):
        return len(self.metadata)
 
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["uid"]
        audio_path = os.path.join(self.audio_dir, f"{uid}.mp3")
        task_row = self.task.iloc[idx]
 
        # Retrieve label only if not a test dataset
        label = None
        if not self.test and self.labels is not None:
            label_row = self.labels[self.labels["uid"] == uid]
            if not label_row.empty:
                label = label_row["label_encoded"].values[0]
 
        # Extract age and gender from metadata
        age = row["age"]
        gender = 0 if row["gender"] == "male" else 1
        task_name = self.task_map[task_row['task']]
        language = self.lang_map[task_row['lang']]
        nltk_features = task_row[['stop_words','stemmed','lemmed','num_total_words','num_unique_words']]
        nltk_features = [x/task_row['audio_length'] for x in nltk_features]
        nltk_features = torch.tensor(nltk_features, dtype=torch.float32)
 
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
 
        if waveform.shape[1] > self.max_length_samples:
            waveform = waveform[:, :self.max_length_samples]
        else:
            padding = self.max_length_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
 
        if self.augment:
            waveform = augmentations(waveform.unsqueeze(0), sample_rate=self.target_sr).squeeze(0)
 
        # Process audio with the Wav2Vec2 processor
        inputs = self.processor(waveform.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt", padding=True)
 
        # Get features
        feature_rows = self.features[self.features["uid"] == uid]
        features = torch.tensor(feature_rows.iloc[:, 3:].values, dtype=torch.float32)
 
        # Process text using BERT tokenizer
        text_row = self.text_data[self.text_data["uid"] == uid]
        if not text_row.empty:
            text = text_row["translation"].values[0]
        else:
            text = ""  # Default empty string if not found
        text_inputs = bert_tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
 
        return {
            "uid": uid,
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long) if label is not None else None,
            "age": torch.tensor(age, dtype=torch.float),
            "gender": torch.tensor(gender, dtype=torch.float),
            "features": features,
            "text_input_ids": text_inputs["input_ids"].squeeze(0),
            "text_attention_mask": text_inputs["attention_mask"].squeeze(0),
            "task": torch.tensor(task_name, dtype=torch.float),
            "lang": torch.tensor(language, dtype=torch.float),
            "nlkt": nltk_features
        }