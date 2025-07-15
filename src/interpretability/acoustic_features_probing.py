import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
from pathlib import Path
from collections import defaultdict
import glob
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from scipy.signal import hilbert, find_peaks, butter, filtfilt
from scipy.stats import entropy
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


class MultiSampleAcousticProbing:

    
    def __init__(self, model, processor, device='cuda', random_state=42):
        self.model = model
        self.processor = processor
        self.device = device
        self.random_state = random_state
        
        # Class names and mapping
        self.class_names = ['Control', 'MCI', 'ADRD', 'Other']
        self.diagnosis_mapping = {'hc': 0, 'mci': 1, 'ad': 2, 'other_dem': 3}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Storage for batch processing
        self.layer_representations = {}
        self.acoustic_features = {}
        self.metadata = {}
        
        # Target layers for analysis (sample every 4th layer)
        self.target_layers = [f'encoder.layers.{i}' for i in [0, 4, 8, 12, 16, 20, 24, 28, 31]]
        
        print(f"Initialized  probing for {len(self.target_layers)} layers")
        print(f"Target layers: {[int(l.split('.')[-1]) for l in self.target_layers]}")
        
    def load_metadata_from_csv(self, csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded metadata for {len(df)} files from {csv_path}")
            print(f"Columns: {list(df.columns)}")
            print(f"Diagnosis distribution: {df['diagnosis'].value_counts().to_dict()}")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def match_audio_to_metadata(self, audio_files: List[str], metadata_df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
        """Match audio files to metadata entries"""
        matched_files = []
        matched_metadata = []
        
        # Create lookup by unique_id 
        metadata_lookup = {}
        for _, row in metadata_df.iterrows():
            unique_id = str(row['unique_id'])
            metadata_lookup[f"{unique_id}.mp3"] = row.to_dict()
        
        print(f"Created lookup for {len(metadata_lookup)} metadata entries")
        
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            
            if filename in metadata_lookup:
                row_data = metadata_lookup[filename]
                
                diagnosis = row_data.get('diagnosis', 'unknown')
                if diagnosis in self.diagnosis_mapping:
                    class_idx = self.diagnosis_mapping[diagnosis]
                    class_name = self.class_names[class_idx]
                    
                    metadata = {
                        'valid': True,
                        'id': row_data.get('id', filename),
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'diagnosis': diagnosis,
                        'task': row_data.get('task', 'unknown'),
                        'language': row_data.get('language', 'unknown'),
                        'study': row_data.get('study', 'unknown'),
                        'unique_id': row_data.get('unique_id', ''),
                        'filename': filename,
                        'file_path': audio_file
                    }
                    
                    matched_files.append(audio_file)
                    matched_metadata.append(metadata)
        
        print(f"Successfully matched {len(matched_files)} audio files to metadata")
        if len(matched_files) > 0:
            class_dist = pd.Series([m['class_name'] for m in matched_metadata]).value_counts()
            print(f"Class distribution: {class_dist.to_dict()}")
            
        return matched_files, matched_metadata
    
    def extract_acoustic_features(self, audio_path: str) -> Dict:
    
        try:
            print(f"    Loading audio: {os.path.basename(audio_path)}")
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Ensure minimum length
            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                print(f"    Audio too short: {audio_path}")
                return {}
            
            # Limit length for speed
            if len(y) > sr * 30:  # Limit to 30 seconds
                y = y[:sr * 30]
            
            features = {}
            print(f"    Audio length: {len(y)/sr:.1f}s")
            
            # ============ BASIC ENERGY FEATURES ============ #
            print(f"    Extracting energy features...")
            
            # RMS Energy
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_range'] = float(np.max(rms) - np.min(rms))
            features['rms_skewness'] = float(stats.skew(rms))
            features['rms_kurtosis'] = float(stats.kurtosis(rms))
            
            # Peak-to-RMS ratio 
            features['peak_to_rms_ratio'] = float(np.max(np.abs(y)) / (np.mean(rms) + 1e-10))
            
            # ============ ZERO CROSSING FEATURES ============
            print(f"    Extracting ZCR features...")
            
            # 2. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            features['zcr_range'] = float(np.max(zcr) - np.min(zcr))
            
            # ============ SPECTRAL FEATURES ============
            print(f"    Extracting spectral features...")
            
            # 3. Spectral shape features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
            features['spec_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spec_centroid_std'] = float(np.std(spectral_centroids))
            features['spec_centroid_range'] = float(np.max(spectral_centroids) - np.min(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
            features['spec_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spec_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
            features['spec_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spec_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Spectral contrast
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
            features['spec_contrast_mean'] = float(np.mean(spec_contrast))
            features['spec_contrast_std'] = float(np.std(spec_contrast))
            
            # Spectral flatness
            spec_flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
            features['spec_flatness_mean'] = float(np.mean(spec_flatness))
            features['spec_flatness_std'] = float(np.std(spec_flatness))
            
            # ============ F0 FEATURES ============
            print(f"    Extracting F0 features...")
            
            # 4. F0 analysis 
            try:
                f0_yin = librosa.yin(y, fmin=50, fmax=400, hop_length=512)
                f0_valid = f0_yin[~np.isnan(f0_yin) & ~np.isinf(f0_yin) & (f0_yin > 0)]
                
                if len(f0_valid) > 10:
                    features['f0_mean'] = float(np.mean(f0_valid))
                    features['f0_std'] = float(np.std(f0_valid))
                    features['f0_range'] = float(np.max(f0_valid) - np.min(f0_valid))
                    features['f0_voiced_ratio'] = len(f0_valid) / len(f0_yin)
                    
                    # F0 contour features
                    f0_diff = np.diff(f0_valid)
                    features['f0_contour_std'] = float(np.std(f0_diff))
                    features['f0_jitter_approx'] = float(np.mean(np.abs(f0_diff) / f0_valid[:-1]))

            except Exception as e:
                print(f"    F0 extraction failed: {e}")

            
            # ============ VOICE QUALITY  ============
            print(f"    Extracting voice quality features...")
            
            # 5. HNR approximation 
            try:
                # Autocorrelation-based HNR estimate
                autocorr = np.correlate(y, y, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                
                if len(autocorr) > 100:
                    min_idx = np.argmin(autocorr[20:100]) + 20
                    features['hnr_approx'] = float(-20 * np.log10(abs(autocorr[min_idx]) + 1e-10))
                else:
                    features['hnr_approx'] = 15.0
                    
                # Shimmer approximation
                analytic_signal = hilbert(y)
                amplitude_envelope = np.abs(analytic_signal)
                if len(amplitude_envelope) > 1:
                    amp_diff = np.diff(amplitude_envelope)
                    features['shimmer_approx'] = float(np.mean(np.abs(amp_diff) / (amplitude_envelope[:-1] + 1e-10)))
                else:
                    features['shimmer_approx'] = 0.05
                    
            except Exception as e:
                print(f"    Voice quality failed: {e}")
                features['hnr_approx'] = 15.0
                features['shimmer_approx'] = 0.05
            
            # ============ TEMPORAL FEATURES ============
            print(f"    Extracting temporal features...")
            
            # 6. Temporal dynamics 
            n_segments = 3
            segment_length = len(y) // n_segments
            
            segment_energies = []
            segment_zcrs = []
            
            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length if i < n_segments - 1 else len(y)
                segment = y[start_idx:end_idx]
                
                if len(segment) > 0:
                    # Energy per segment
                    segment_rms = np.sqrt(np.mean(segment**2))
                    features[f'rms_seg{i+1}'] = float(segment_rms)
                    segment_energies.append(segment_rms)
                    
                    # ZCR per segment
                    segment_zcr = librosa.feature.zero_crossing_rate(segment)[0]
                    features[f'zcr_seg{i+1}'] = float(np.mean(segment_zcr))
                    segment_zcrs.append(np.mean(segment_zcr))
            
            # Energy variability
            if segment_energies:
                features['energy_variability'] = float(np.std(segment_energies) / (np.mean(segment_energies) + 1e-10))
                features['zcr_variability'] = float(np.std(segment_zcrs) / (np.mean(segment_zcrs) + 1e-10))
            
            # 7. Speech rate
            try:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, delta=0.1)
                features['speech_rate'] = len(onset_frames) / (len(y) / sr)
                
                if len(onset_frames) > 1:
                    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                    ioi = np.diff(onset_times)
                    features['rhythm_regularity'] = float(1.0 / (np.std(ioi) + 1e-10))
                else:
                    features['rhythm_regularity'] = 1.0
            except:
                features['speech_rate'] = 5.0
                features['rhythm_regularity'] = 1.0
            
            # ============ MFCC FEATURES (CORE ONLY) ============
            print(f"    Extracting MFCC features...")
            
            # 8. MFCC (first 6 coefficients)
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6, hop_length=512)
                
                for i in range(6):
                    features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                    features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
                
                # Delta MFCCs (first 3 only)
                mfcc_delta = librosa.feature.delta(mfccs[:3])
                for i in range(3):
                    features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
                    
            except Exception as e:
                print(f"    MFCC failed: {e}")
                for i in range(6):
                    features[f'mfcc_{i+1}_mean'] = 0.0
                    features[f'mfcc_{i+1}_std'] = 1.0
                for i in range(3):
                    features[f'mfcc_delta_{i+1}_mean'] = 0.0
            
            # ============ SPECTRAL DYNAMICS ============
            print(f"    Extracting spectral dynamics...")
            
            # 9. Spectral stability
            try:
                stft = librosa.stft(y, hop_length=512, n_fft=1024)  # Smaller FFT
                mag_spec = np.abs(stft)
                
                if mag_spec.shape[1] > 1:
                    # Spectral flux
                    spec_diff = np.diff(mag_spec, axis=1)
                    features['spectral_flux_mean'] = float(np.mean(np.sum(spec_diff**2, axis=0)))
                    
                    # Spectral stability (sample every 5th frame)
                    spec_corr = []
                    for i in range(0, mag_spec.shape[1] - 5, 5):
                        corr = np.corrcoef(mag_spec[:, i], mag_spec[:, i+5])[0, 1]
                        if not np.isnan(corr):
                            spec_corr.append(corr)
                    
                    if spec_corr:
                        features['spectral_stability'] = float(np.mean(spec_corr))
                    else:
                        features['spectral_stability'] = 0.8
                else:
                    features['spectral_flux_mean'] = 0.1
                    features['spectral_stability'] = 0.8
            except:
                features['spectral_flux_mean'] = 0.1
                features['spectral_stability'] = 0.8
            
            # ============ PAUSE ANALYSIS  ============
            print(f"    Extracting pause features...")
            
            # 10.  Pause detection
            try:
                energy_threshold = np.percentile(rms, 30)  # Bottom 30% as silences
                silent_frames = rms < energy_threshold
                
                # Count transitions to estimate pauses
                transitions = np.sum(np.diff(silent_frames.astype(int)) != 0)
                features['pause_rate_approx'] = transitions / (len(y) / sr)
                
                # Estimate pause ratio
                silence_ratio = np.mean(silent_frames)
                features['silence_ratio'] = float(silence_ratio)
                
            except:
                features['pause_rate_approx'] = 1.0
                features['silence_ratio'] = 0.2
            
            print(f"    Extracted {len(features)} features successfully!")
            return features
            
        except Exception as e:
            print(f"    ERROR extracting features from {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_layer_representation(self, audio_path: str, layer_name: str) -> np.ndarray:
        """Extract representation from specific layer"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Limit length to prevent memory issues
            if len(y) > 16000 * 60:  # 1 minute max
                y = y[:16000 * 60]
                
            inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            
            # Hook to capture representation
            activation = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activation['output'] = output[0].detach()
                else:
                    activation['output'] = output.detach()
            
            # Get layer and register hook
            layer = self._get_layer_by_name(layer_name)
            if layer is None:
                print(f"Layer {layer_name} not found")
                return np.array([])
                
            handle = layer.register_forward_hook(hook_fn)
            
            # Forward pass
            with torch.no_grad():
                if hasattr(self.model, 'module'):
                    self.model.module(input_features)
                else:
                    self.model(input_features)
            
            handle.remove()
            
            if 'output' in activation:
                # Global average pooling over time dimension
                representation = activation['output'].mean(dim=1).squeeze().cpu().numpy()
                return representation
            else:
                print(f"No activation captured for layer {layer_name}")
                return np.array([])
                
        except Exception as e:
            print(f"Error extracting representation from {audio_path}, layer {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
    
    def _get_layer_by_name(self, name: str):
        """Get layer by name from the model"""
        parts = name.split('.')
        layer = self.model
        
        # Handle DataParallel wrapper
        if hasattr(layer, 'module'):
            layer = layer.module
            
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                return None
        return layer
    
    def batch_extract_features_and_representations(self, audio_files: List[str], 
                                                  metadata: List[Dict],
                                                  batch_size: int = 16) -> Tuple[Dict, List, List]:

        print("Extracting streamlined acoustic features and neural representations...")
        
        valid_files = []
        valid_metadata = []
        all_acoustic_features = []
        layer_representations = {layer: [] for layer in self.target_layers}
        
        total_files = len(audio_files)
        for i, (audio_file, file_metadata) in enumerate(zip(audio_files, metadata)):
            
            print(f"\n{'='*60}")
            print(f"Processing {i+1}/{total_files}: {os.path.basename(audio_file)}")
            print(f"Class: {file_metadata['class_name']}, Task: {file_metadata.get('task', 'unknown')}")
            
            # Extract acoustic features
            print("Step 1/2: Extracting acoustic features...")
            acoustic_features = self.extract_acoustic_features(audio_file)
            if not acoustic_features:
                print(f"❌ Failed to extract acoustic features for {audio_file}")
                continue
                
            print(f"✅ Extracted {len(acoustic_features)} acoustic features")
            
            # Extract neural representations for each layer
            print("Step 2/2: Extracting neural representations...")
            layer_reprs = {}
            skip_file = False
            
            for j, layer_name in enumerate(self.target_layers):
                print(f"    Extracting layer {layer_name} ({j+1}/{len(self.target_layers)})...")
                repr_vec = self.extract_layer_representation(audio_file, layer_name)
                if len(repr_vec) == 0:
                    print(f"    ❌ Failed to extract representation for layer {layer_name}")
                    skip_file = True
                    break
                layer_reprs[layer_name] = repr_vec
                print(f"    ✅ Layer {layer_name}: shape {repr_vec.shape}")
            
            if skip_file:
                print(f"❌ Skipping file due to representation extraction failure")
                continue
                
            # Store all data
            valid_files.append(audio_file)
            valid_metadata.append(file_metadata)
            all_acoustic_features.append(acoustic_features)
            
            for layer_name in self.target_layers:
                layer_representations[layer_name].append(layer_reprs[layer_name])
            
            print(f"✅ Successfully processed {os.path.basename(audio_file)}")
            print(f"Progress: {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
        
        processed_layer_reprs = {}
        for layer_name in self.target_layers:
            if layer_representations[layer_name]:
                processed_layer_reprs[layer_name] = np.vstack(layer_representations[layer_name])
                print(f"Final layer {layer_name}: shape {processed_layer_reprs[layer_name].shape}")
            else:
                processed_layer_reprs[layer_name] = np.array([])
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Successfully processed: {len(valid_files)}/{total_files} files")
        print(f"Feature count per file: {len(all_acoustic_features[0]) if all_acoustic_features else 0}")
        print(f"Class distribution: {dict(pd.Series([m['class_name'] for m in valid_metadata]).value_counts())}")
        
        return processed_layer_reprs, all_acoustic_features, valid_metadata
    
    def create_control_features(self, metadata: List[Dict], audio_files: List[str]) -> Dict:
        """Create control features that should NOT be encoded in representations"""
        control_features = {}
        
        # File-based controls (should not be predictable from neural representations)
        control_features['file_size'] = [os.path.getsize(f) for f in audio_files]
        control_features['filename_length'] = [len(os.path.basename(f)) for f in audio_files]
        control_features['alphabetic_order'] = [ord(os.path.basename(f)[0]) for f in audio_files]
        
        # Random controls
        np.random.seed(self.random_state)
        control_features['random_continuous'] = np.random.normal(0, 1, len(audio_files))
        control_features['random_binary'] = np.random.choice([0, 1], len(audio_files))
        
        return control_features
    
    def perform_probing_analysis(self, layer_representations: Dict, acoustic_features: List[Dict],
                               metadata: List[Dict], audio_files: List[str],
                               cv_folds: int = 5) -> Dict:

        print("Performing probing analysis with cross-validation...")
        
        # Prepare acoustic feature matrix
        if not acoustic_features:
            raise ValueError("No acoustic features available")
            
        feature_names = list(acoustic_features[0].keys())
        acoustic_matrix = np.array([[sample.get(feat, 0) for feat in feature_names] 
                                  for sample in acoustic_features])
        
        # Handle NaN values
        acoustic_matrix = np.nan_to_num(acoustic_matrix)
        
        print(f"Acoustic feature matrix shape: {acoustic_matrix.shape}")
        print(f"Features: {feature_names[:10]}...")  # Show first 10
        
        # Create control features
        control_features = self.create_control_features(metadata, audio_files)
        
        labels = [meta['class_idx'] for meta in metadata]
        n_samples = len(labels)
        
        if n_samples < 10:
            cv_folds = min(3, n_samples)  # Use 3-fold or fewer for very small samples
            print(f"Small sample size ({n_samples}), using {cv_folds}-fold CV")
        elif n_samples < 20:
            cv_folds = min(4, cv_folds)
            print(f"Moderate sample size ({n_samples}), using {cv_folds}-fold CV")
        
        # Initialize results storage
        probing_results = {}
        
        # Perform probing for each layer
        for layer_name in tqdm(self.target_layers, desc="Probing layers"):
            if layer_name not in layer_representations or len(layer_representations[layer_name]) == 0:
                print(f"Skipping layer {layer_name}: no representations")
                continue
                
            layer_matrix = layer_representations[layer_name]
            print(f"Layer {layer_name} matrix shape: {layer_matrix.shape}")
            layer_results = {}
            
            scaler = StandardScaler()
            layer_matrix_scaled = scaler.fit_transform(layer_matrix)
            
            # Probe each acoustic feature
            for i, feat_name in enumerate(feature_names):
                target = acoustic_matrix[:, i]
                
                # Skip if no variation
                if np.std(target) == 0:
                    print(f"  Skipping {feat_name}: no variation")
                    continue
                
                # Regression probing 
                reg_scores = self._probe_continuous_target(
                    layer_matrix_scaled, target, labels, cv_folds
                )
                
                # Classification probing (binary targets)
                median_threshold = np.median(target)
                binary_target = (target > median_threshold).astype(int)
                
                # Check if binary split is valid (at least 2 samples of each class)
                unique_classes, class_counts = np.unique(binary_target, return_counts=True)
                min_class_size = np.min(class_counts)
                
                if len(unique_classes) < 2 or min_class_size < 2:
                    print(f"  Skipping {feat_name}: binary split invalid (classes: {dict(zip(unique_classes, class_counts))})")
                    # Store regression results only
                    layer_results[feat_name] = {
                        'regression_r2': reg_scores['mean_r2'],
                        'regression_r2_std': reg_scores['std_r2'],
                        'classification_acc': 0.5,  # Chance level
                        'classification_acc_std': 0.0,
                        'baseline_acc': 0.5,
                        'effect_size': 0.0,
                        'n_samples': len(target),
                        'binary_split_failed': True
                    }
                    continue
                
                clf_scores = self._probe_binary_target(
                    layer_matrix_scaled, binary_target, labels, cv_folds
                )
                
                layer_results[feat_name] = {
                    'regression_r2': reg_scores['mean_r2'],
                    'regression_r2_std': reg_scores['std_r2'],
                    'classification_acc': clf_scores['mean_acc'],
                    'classification_acc_std': clf_scores['std_acc'],
                    'baseline_acc': clf_scores['baseline'],
                    'effect_size': clf_scores['mean_acc'] - clf_scores['baseline'],
                    'n_samples': len(target),
                    'binary_split_failed': False
                }
            
            # Probe control features
            layer_results['_controls'] = {}
            for control_name, control_target in control_features.items():
                control_target = np.array(control_target)
                
                if control_name.endswith('_binary') or len(np.unique(control_target)) == 2:
                    # Binary control - check validity
                    unique_classes, class_counts = np.unique(control_target, return_counts=True)
                    if len(unique_classes) >= 2 and np.min(class_counts) >= 2:
                        try:
                            control_scores = self._probe_binary_target(
                                layer_matrix_scaled, control_target, labels, cv_folds
                            )
                            layer_results['_controls'][control_name] = control_scores['mean_acc']
                        except:
                            layer_results['_controls'][control_name] = 0.5
                    else:
                        layer_results['_controls'][control_name] = 0.5
                else:
                    # Continuous control
                    try:
                        control_scores = self._probe_continuous_target(
                            layer_matrix_scaled, control_target, labels, cv_folds
                        )
                        layer_results['_controls'][control_name] = control_scores['mean_r2']
                    except:
                        layer_results['_controls'][control_name] = 0.0
            
            probing_results[layer_name] = layer_results
            valid_features = len([f for f in layer_results.keys() if not f.startswith('_')])
            print(f"Completed probing for layer {layer_name}: {valid_features} valid features")
        
        #  statistical significance testing 
        if n_samples >= 10:
            probing_results = self._add_statistical_testing(probing_results)
        else:
            print(f"Skipping significance testing: sample size ({n_samples}) too small")
        
        return probing_results
    
    def _probe_continuous_target(self, representations: np.ndarray, target: np.ndarray,
                               labels: List[int], cv_folds: int) -> Dict:
        
        n_samples = len(target)
        
        if n_samples < 10:
            try:
                split_idx = int(0.7 * n_samples)
                indices = np.random.permutation(n_samples)
                train_idx, test_idx = indices[:split_idx], indices[split_idx:]
                
                if len(train_idx) == 0 or len(test_idx) == 0:
                    train_idx = test_idx = np.arange(n_samples)
                
                X_train, X_test = representations[train_idx], representations[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]
                
                regressor = Ridge(alpha=1.0, random_state=self.random_state)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                return {
                    'mean_r2': max(0, r2),  # Floor at 0
                    'std_r2': 0.0,
                    'scores': [max(0, r2)]
                }
                
            except Exception as e:
                print(f"    Simple regression split failed: {e}")
                return {
                    'mean_r2': 0.0,
                    'std_r2': 0.0,
                    'scores': [0.0]
                }
        
        # For larger samples, use cross-validation
        try:
            # Use stratified CV if possible, otherwise regular CV
            try:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                cv_iterator = skf.split(representations, labels)
            except ValueError:
                # Fall back to regular KFold if stratified fails
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                cv_iterator = kf.split(representations)
            
            r2_scores = []
            
            for train_idx, test_idx in cv_iterator:
                X_train, X_test = representations[train_idx], representations[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]
                
                regressor = Ridge(alpha=1.0, random_state=self.random_state)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(max(0, r2))  # Floor at 0
            
            return {
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'scores': r2_scores
            }
            
        except Exception as e:
            print(f"    Regression CV failed: {e}")
            return {
                'mean_r2': 0.0,
                'std_r2': 0.0,
                'scores': [0.0]
            }
    
    def _probe_binary_target(self, representations: np.ndarray, target: np.ndarray,
                           labels: List[int], cv_folds: int) -> Dict:
        
        n_samples = len(target)
        
        if n_samples < 10:
            try:
                split_idx = int(0.7 * n_samples)
                indices = np.random.permutation(n_samples)
                train_idx, test_idx = indices[:split_idx], indices[split_idx:]
                
                if len(train_idx) == 0 or len(test_idx) == 0:
                    train_idx = test_idx = np.arange(n_samples)
                
                X_train, X_test = representations[train_idx], representations[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]
                
                # Check if we have both classes in training
                unique_train = np.unique(y_train)
                if len(unique_train) < 2:
                    # Fallback: perfect classifier based on majority class
                    baseline = max(np.mean(target), 1 - np.mean(target))
                    return {
                        'mean_acc': baseline,
                        'std_acc': 0.0,
                        'baseline': baseline,
                        'scores': [baseline]
                    }
                
                classifier = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                )
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                baseline = max(np.mean(target), 1 - np.mean(target))
                return {
                    'mean_acc': acc,
                    'std_acc': 0.0,
                    'baseline': baseline,
                    'scores': [acc]
                }
                
            except Exception as e:
                print(f"    Simple split failed: {e}")
                baseline = max(np.mean(target), 1 - np.mean(target))
                return {
                    'mean_acc': baseline,
                    'std_acc': 0.0,
                    'baseline': baseline,
                    'scores': [baseline]
                }
        
        try:
            try:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                cv_iterator = skf.split(representations, target)
            except ValueError:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                cv_iterator = kf.split(representations)
            
            accuracies = []
            baseline = max(np.mean(target), 1 - np.mean(target))
            
            for train_idx, test_idx in cv_iterator:
                X_train, X_test = representations[train_idx], representations[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]
                
                unique_train = np.unique(y_train)
                if len(unique_train) < 2:
                    accuracies.append(baseline)
                    continue
                
                classifier = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                )
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)
            
            if not accuracies:
                return {
                    'mean_acc': baseline,
                    'std_acc': 0.0,
                    'baseline': baseline,
                    'scores': [baseline]
                }
            
            return {
                'mean_acc': np.mean(accuracies),
                'std_acc': np.std(accuracies),
                'baseline': baseline,
                'scores': accuracies
            }
            
        except Exception as e:
            print(f"    CV failed: {e}")
            baseline = max(np.mean(target), 1 - np.mean(target))
            return {
                'mean_acc': baseline,
                'std_acc': 0.0,
                'baseline': baseline,
                'scores': [baseline]
            }
    
    def _add_statistical_testing(self, probing_results: Dict) -> Dict:
        """Add statistical significance testing with multiple comparison correction"""
        
        # Collect all p-values for multiple comparison correction
        all_p_values = []
        p_value_keys = []
        
        for layer_name, layer_results in probing_results.items():
            for feat_name, feat_results in layer_results.items():
                if feat_name.startswith('_'):  # Skip control features for main analysis
                    continue
                    
                # One-sample t-test against baseline (chance performance)
                if 'classification_acc' in feat_results:
                    baseline = feat_results['baseline_acc']
                    effect_size = feat_results['effect_size']
                    std_err = feat_results['classification_acc_std']
                    n_samples = feat_results['n_samples']
                    
                    # t-statistic for testing if mean > baseline
                    if std_err > 0:
                        t_stat = effect_size / (std_err / np.sqrt(5))  # 5-fold CV
                        p_value = stats.t.sf(t_stat, df=4)  # One-tailed test
                    else:
                        p_value = 0.5
                    
                    feat_results['p_value'] = p_value
                    all_p_values.append(p_value)
                    p_value_keys.append((layer_name, feat_name))
        
        # Multiple comparison correction (Bonferroni)
        if all_p_values:
            corrected_p_values = multipletests(all_p_values, method='bonferroni')[1]
            
            for i, (layer_name, feat_name) in enumerate(p_value_keys):
                probing_results[layer_name][feat_name]['p_value_corrected'] = corrected_p_values[i]
                probing_results[layer_name][feat_name]['significant'] = corrected_p_values[i] < 0.05
        
        return probing_results
    
    def visualize_probing_results(self, probing_results: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        self._plot_probing_heatmap(probing_results, output_dir, metric='classification_acc')
        
        self._plot_probing_heatmap(probing_results, output_dir, metric='regression_r2')
        
        self._plot_significance_heatmap(probing_results, output_dir)
        
        self._plot_layer_trends(probing_results, output_dir)
        
        self._plot_control_analysis(probing_results, output_dir)
        
        self._plot_feature_importance(probing_results, output_dir)
        
    def _plot_probing_heatmap(self, probing_results: Dict, output_dir: str, metric: str):
        
        layers = list(probing_results.keys())
        if not layers:
            return
            
        # Get all features (exclude controls)
        all_features = set()
        for layer_results in probing_results.values():
            for feat_name in layer_results.keys():
                if not feat_name.startswith('_'):
                    all_features.add(feat_name)
        
        features = sorted(list(all_features))
        
        if not features:
            return
            
        # Create matrix
        matrix = np.zeros((len(features), len(layers)))
        
        for i, feature in enumerate(features):
            for j, layer in enumerate(layers):
                if feature in probing_results[layer]:
                    value = probing_results[layer][feature].get(metric, 0)
                    matrix[i, j] = value
        
        # Plot
        plt.figure(figsize=(14, max(8, len(features) * 0.3)))
        
        if metric == 'classification_acc':
            vmin, vmax = 0.4, 1.0
            cmap = 'RdYlGn'
            cbar_label = 'Classification Accuracy'
            title = ' Acoustic Feature Probing: Classification Accuracy'
        else:
            vmin, vmax = 0.0, 0.8
            cmap = 'viridis'
            cbar_label = 'Regression R²'
            title = ' Acoustic Feature Probing: Regression R²'
        
        im = plt.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        layer_labels = [f"L{layer.split('.')[-1]}" for layer in layers]
        plt.xticks(range(len(layers)), layer_labels)
        plt.yticks(range(len(features)), features, fontsize=8)
        plt.xlabel('Transformer Layer')
        plt.ylabel('Acoustic Feature')
        plt.title(title + f'\n{len(features)} features × {len(layers)} layers')
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        
        if metric == 'classification_acc':
            plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
            plt.text(len(layers)-1, -0.3, 'Chance (0.5)', color='red', ha='right')
        
        for i in range(len(features)):
            for j in range(len(layers)):
                value = matrix[i, j]
                if (metric == 'classification_acc' and value > 0.7) or \
                   (metric == 'regression_r2' and value > 0.3):
                    color = 'white' if value > 0.6 else 'black'
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=color, fontweight='bold', fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'_probing_heatmap_{metric}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_significance_heatmap(self, probing_results: Dict, output_dir: str):
        
        layers = list(probing_results.keys())
        features = []
        
        # Get features that have significance testing
        for layer_results in probing_results.values():
            for feat_name in layer_results.keys():
                if not feat_name.startswith('_') and feat_name not in features:
                    if 'p_value_corrected' in layer_results[feat_name]:
                        features.append(feat_name)
        
        if not features:
            return
            
        # Create significance matrix
        sig_matrix = np.zeros((len(features), len(layers)))
        
        for i, feature in enumerate(features):
            for j, layer in enumerate(layers):
                if feature in probing_results[layer]:
                    p_val = probing_results[layer][feature].get('p_value_corrected', 1.0)
                    # Convert p-value to significance level
                    if p_val < 0.001:
                        sig_matrix[i, j] = 3  # ***
                    elif p_val < 0.01:
                        sig_matrix[i, j] = 2  # **
                    elif p_val < 0.05:
                        sig_matrix[i, j] = 1  # *
                    else:
                        sig_matrix[i, j] = 0  # n.s.
        
        plt.figure(figsize=(12, max(6, len(features) * 0.2)))
        
        colors = ['lightgray', 'lightblue', 'orange', 'red']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        im = plt.imshow(sig_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=3)
        
        layer_labels = [f"L{layer.split('.')[-1]}" for layer in layers]
        plt.xticks(range(len(layers)), layer_labels)
        plt.yticks(range(len(features)), features, fontsize=8)
        plt.xlabel('Transformer Layer')
        plt.ylabel('Acoustic Feature')
        plt.title('Statistical Significance of  Probing Results\n(Bonferroni corrected)')
        
        legend_elements = [
            mpatches.Patch(color='lightgray', label='n.s. (p ≥ 0.05)'),
            mpatches.Patch(color='lightblue', label='* (p < 0.05)'),
            mpatches.Patch(color='orange', label='** (p < 0.01)'),
            mpatches.Patch(color='red', label='*** (p < 0.001)')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '_probing_significance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_layer_trends(self, probing_results: Dict, output_dir: str):
        
        layers = list(probing_results.keys())
        layer_nums = [int(layer.split('.')[-1]) for layer in layers]
        
        feature_groups = {
            'Energy & Amplitude': ['rms_mean', 'rms_std', 'rms_range', 'rms_skewness', 'rms_kurtosis',
                                  'peak_to_rms_ratio'] + [f'rms_seg{i}' for i in range(1, 4)] + ['energy_variability'],
            'Spectral Shape': ['spec_centroid_mean', 'spec_centroid_std', 'spec_centroid_range',
                             'spec_bandwidth_mean', 'spec_bandwidth_std', 
                             'spec_rolloff_mean', 'spec_rolloff_std',
                             'spec_contrast_mean', 'spec_contrast_std',
                             'spec_flatness_mean', 'spec_flatness_std'],
            'Spectral Dynamics': ['spectral_flux_mean', 'spectral_stability'],
            'Pitch & F0': ['f0_mean', 'f0_std', 'f0_range', 'f0_voiced_ratio', 
                          'f0_contour_std', 'f0_jitter_approx'],
            'Voice Quality': ['hnr_approx', 'shimmer_approx'],
            'Voicing & ZCR': ['zcr_mean', 'zcr_std', 'zcr_range'] + [f'zcr_seg{i}' for i in range(1, 4)] + ['zcr_variability'],
            'MFCC': [f'mfcc_{i}_mean' for i in range(1, 7)] + [f'mfcc_{i}_std' for i in range(1, 7)] + 
                   [f'mfcc_delta_{i}_mean' for i in range(1, 4)],
            'Temporal & Rhythm': ['speech_rate', 'rhythm_regularity'],
            'Pauses & Silence': ['pause_rate_approx', 'silence_ratio']
        }
        
        plt.figure(figsize=(16, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_groups)))
        
        for group_idx, (group_name, group_features) in enumerate(feature_groups.items()):
            group_accuracies = []
            
            for layer in layers:
                layer_results = probing_results[layer]
                layer_accs = []
                
                for feat in group_features:
                    if feat in layer_results:
                        acc = layer_results[feat].get('classification_acc', 0)
                        layer_accs.append(acc)
                
                if layer_accs:
                    group_accuracies.append(np.mean(layer_accs))
                else:
                    group_accuracies.append(0)
            
            plt.plot(layer_nums, group_accuracies, 'o-', label=group_name, 
                    color=colors[group_idx], linewidth=2, markersize=6)
        
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
        plt.xlabel('Transformer Layer')
        plt.ylabel('Mean Classification Accuracy')
        plt.title(' Feature Type Encoding Across Layers')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '_layer_trends.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_control_analysis(self, probing_results: Dict, output_dir: str):
        
        layers = list(probing_results.keys())
        layer_nums = [int(layer.split('.')[-1]) for layer in layers]
        
        # Extract control results
        control_results = {}
        for layer in layers:
            if '_controls' in probing_results[layer]:
                for control_name, control_acc in probing_results[layer]['_controls'].items():
                    if control_name not in control_results:
                        control_results[control_name] = []
                    control_results[control_name].append(control_acc)
        
        if not control_results:
            return
            
        plt.figure(figsize=(12, 6))
        
        for control_name, accuracies in control_results.items():
            plt.plot(layer_nums, accuracies, 'o-', label=control_name, alpha=0.7)
        
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
        plt.xlabel('Transformer Layer')
        plt.ylabel('Control Feature Accuracy/R²')
        plt.title('Control Feature Analysis\n(Should remain at chance level)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '_control_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_feature_importance(self, probing_results: Dict, output_dir: str):
        
        feature_scores = {}
        
        for layer_results in probing_results.values():
            for feat_name, feat_results in layer_results.items():
                if not feat_name.startswith('_'):
                    acc = feat_results.get('classification_acc', 0)
                    if feat_name not in feature_scores:
                        feature_scores[feat_name] = []
                    feature_scores[feat_name].append(acc)
        
        feature_stats = {}
        for feat_name, scores in feature_scores.items():
            feature_stats[feat_name] = {
                'mean': np.mean(scores),
                'max': np.max(scores),
                'std': np.std(scores)
            }
        
        # Sort by mean accuracy
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        # Plot top 30 features
        top_features = sorted_features[:30]
        feature_names = [f[0] for f in top_features]
        mean_accs = [f[1]['mean'] for f in top_features]
        max_accs = [f[1]['max'] for f in top_features]
        
        plt.figure(figsize=(14, 10))
        
        x_pos = np.arange(len(feature_names))
        plt.barh(x_pos, mean_accs, alpha=0.7, label='Mean across layers')
        plt.barh(x_pos, max_accs, alpha=0.5, label='Max across layers')
        
        plt.yticks(x_pos, feature_names, fontsize=8)
        plt.xlabel('Classification Accuracy')
        plt.title('Top 30 Most Encoded  Acoustic Features')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, probing_results: Dict, metadata: List[Dict], output_dir: str):
        """Save detailed results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(output_dir, '_probing_results.pkl'), 'wb') as f:
            pickle.dump(probing_results, f)
        
        with open(os.path.join(output_dir, '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Create summary CSV
        summary_data = []
        for layer_name, layer_results in probing_results.items():
            layer_num = int(layer_name.split('.')[-1])
            
            for feat_name, feat_results in layer_results.items():
                if not feat_name.startswith('_'):
                    summary_data.append({
                        'layer': layer_num,
                        'layer_name': layer_name,
                        'feature': feat_name,
                        'classification_accuracy': feat_results.get('classification_acc', 0),
                        'regression_r2': feat_results.get('regression_r2', 0),
                        'effect_size': feat_results.get('effect_size', 0),
                        'p_value': feat_results.get('p_value', 1.0),
                        'p_value_corrected': feat_results.get('p_value_corrected', 1.0),
                        'significant': feat_results.get('significant', False)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, '_probing_summary.csv'), index=False)
        
        print(f" results saved to {output_dir}")
        

def run__multi_sample_probing(train_audio_dir: str = "data/train_audios",
                                    test_audio_dir: str = "data/test_audios",
                                    metadata_csv: str = "data/audios_final_partition_detailed_task.csv",
                                    output_dir: str = "results/_multi_sample_probing",
                                    config_path: str = "config/config.yaml",
                                    max_files: Optional[int] = None):
    """
    Main function to run  multi-sample acoustic probing analysis
    """
    
    print("="*80)
    print("MULTI-SAMPLE ACOUSTIC FEATURE PROBING")
    print("="*80)
    
    from utils.utils import load_config
    from transformers import AutoModelForAudioClassification, AutoProcessor
    import torch.nn as nn
    
    config = load_config(config_path)
    
    gpus = config["training"].get("gpus", None)
    if gpus and gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus.strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_id = config['training']['model_name']
    print(f"Loading model: {model_id}")
    
    model = AutoModelForAudioClassification.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        num_labels=config["training"]["num_labels"]
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = nn.DataParallel(model)
    
    if config['model']['load_checkpoint']:
        checkpoint_path = config["paths"]["final_model"]
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    probing_analyzer = MultiSampleAcousticProbing(model, processor, device)
    
    metadata_df = probing_analyzer.load_metadata_from_csv(metadata_csv)
    if metadata_df.empty:
        raise ValueError(f"Could not load metadata from {metadata_csv}")
    
    audio_files = []
    for audio_dir in [train_audio_dir, test_audio_dir]:
        if os.path.exists(audio_dir):
            files = glob.glob(os.path.join(audio_dir, "*.mp3"))
            audio_files.extend(files)
    
    print(f"Found {len(audio_files)} audio files")
    
    matched_files, matched_metadata = probing_analyzer.match_audio_to_metadata(audio_files, metadata_df)
    
    if not matched_files:
        raise ValueError("No audio files could be matched to metadata")
    
    if max_files:
        matched_files = matched_files[:max_files]
        matched_metadata = matched_metadata[:max_files]
        print(f"Limited to {max_files} files for testing")
    
    layer_representations, acoustic_features, final_metadata = probing_analyzer.batch_extract_features_and_representations(
        matched_files, matched_metadata, batch_size=1  
    )
    
    if not acoustic_features:
        raise ValueError("No acoustic features were successfully extracted")
    
    probing_results = probing_analyzer.perform_probing_analysis(
        layer_representations, acoustic_features, final_metadata, matched_files
    )
    
    probing_analyzer.visualize_probing_results(probing_results, output_dir)
    
    probing_analyzer.save_results(probing_results, final_metadata, output_dir)
    
    
    print(f" multi-sample probing analysis complete. Results saved to: {output_dir}")
    
    return probing_analyzer, probing_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run  multi-sample acoustic probing")
    parser.add_argument("--train_audio_dir", default="data/train_audios")
    parser.add_argument("--test_audio_dir", default="data/test_audios") 
    parser.add_argument("--metadata_csv", default="../data/audios_final_partition_detailed_task.csv")
    parser.add_argument("--output_dir", default="results/_multi_sample_probing")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max_files", type=int, help="Limit number of files for testing")
    
    args = parser.parse_args()
    
    analyzer, results = run__multi_sample_probing(
        train_audio_dir=args.train_audio_dir,
        test_audio_dir=args.test_audio_dir,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        config_path=args.config,
        max_files=args.max_files
    )