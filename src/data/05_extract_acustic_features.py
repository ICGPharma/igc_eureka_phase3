import torch
import torchaudio
import opensmile
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import copy

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def segment_audio(waveform, sample_rate, window_size=0.2, step_size=0.1):

    window_length = int(sample_rate * window_size)
    step_length = int(sample_rate * step_size)
    segments = []
    times = []

    for start in tqdm(range(0, waveform.size(1), step_length)):
        end = start + window_length
        if end > waveform.size(1):
            end = waveform.size(1)
        if round(start/sample_rate,3) == round(end/sample_rate,3):
            continue
        segments.append(waveform[:, start:end])
        times.append((round(start/sample_rate,3), round(end/sample_rate,3)))

    return segments, times

# Function to extract features for each audio file
def extract_features(audio_file, uid, window_size=0.2, step_size=0.1, partition=None):
    """
    Extract features from audio file using openSMILE.
    Args:
        audio_file: Path to the audio file.
        uid: Unique identifier for the audio file.
        window_size: Size of each window in seconds.
        step_size: Overlap step size in seconds.
    Returns:
        DataFrame containing extracted features.
    """
    # Load audio
    waveform, sample_rate = load_audio(audio_file)

    # Segment audio
    segments, times = segment_audio(waveform, sample_rate, window_size, step_size)

    # min_size = int(sample_rate*window_size/4)
    min_size = 9600
    
    # Create an empty DataFrame to hold features
    all_features = []

    # Process each segment
    for i, (segment, (start_sec, end_sec)) in tqdm(enumerate(zip(segments, times)), total=len(segments)):

        # Save the temporary segment as a WAV file for openSMILE
        if segment.shape[1] < min_size:
            segment = torch.cat([segment, torch.zeros((segment.shape[0], min_size-segment.shape[1]))], dim=1)
        
        temp_wav = f"temp_segment_{partition}.wav"
        torchaudio.save(temp_wav, segment, sample_rate)
        
        # Extract features using openSMILE
        features = smile.process_file(temp_wav)
        
        # Add metadata
        features['uid'] = uid
        features['segment_start_sec'] = start_sec
        features['segment_end_sec'] = end_sec
        all_features.append(features)

        # Clean up temporary WAV file
        os.remove(temp_wav)
    
    # Combine features into a single DataFrame
    all_features_df = pd.concat(all_features)
    
    # Reorder columns
    reordered_columns = ['uid', 'segment_start_sec', 'segment_end_sec'] + \
                        [col for col in all_features_df.columns if col not in ['uid', 'segment_start_sec', 'segment_end_sec']]
    return all_features_df[reordered_columns]

# Main function to process all audio files in a directory
def process_audio_directory(input_dir, output_file, partition, window_size=0.2, step_size=0.1):
    """
    Process all audio files in a directory and save features to a single CSV.
    Args:
        input_dir: Directory containing audio files.
        output_file: Path to save the combined feature CSV file.
        window_size: Size of each window in seconds.
        step_size: Overlap step size in seconds.
    """
    audio_files = [f for f in Path(input_dir).glob("*.mp3")]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        header_written = False
        for audio_file in tqdm(audio_files):
            uid = Path(audio_file).stem  # Use the file name (without extension) as the UID

            print(f"Processing {audio_file} with UID: {uid}")
            
            # Extract features for the current audio
            features = extract_features(audio_file, uid, window_size, step_size, partition)
            
            # Write to file
            if not header_written:
                features.to_csv(f, index=False)  # Write header for the first file
                header_written = True
            else:
                features.to_csv(f, index=False, header=False)
    
    features = pd.read_csv(output_file)
    features = features.sort_values(by=['uid','segment_start_sec']).reset_index(drop=True)
    features.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Extract features from audio files in a directory.")
    args.add_argument("--processing_partition", type=str, default="train_audios", help="Directory containing audio files. Either train or test partition (train_audios or test_audios).")
    args.add_argument("--path_to_partition", type=str, default="../../data/processed/post_data/", help="Directory containing audio files. Either train or test partition.")
    args.add_argument("--saving_path", type=str, default='../../data/processed/', help="Directory to save the obtained file.")
    args = args.parse_args()

    args.part = copy.deepcopy(args.processing_partition)
    args.output_csv_file = os.path.join(args.saving_path, f"{args.processing_partition}_acustic_features.csv")
    args.processing_partition = os.path.join(args.path_to_partition , args.processing_partition)
    print(f"Input Folder: {args.processing_partition}")
    print(f"Output File: {args.output_csv_file}")
    process_audio_directory(args.processing_partition, args.output_csv_file, args.part)

