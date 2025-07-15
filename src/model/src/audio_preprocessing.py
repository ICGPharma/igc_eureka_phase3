import os
import argparse
import numpy as np
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
import pyloudnorm as pyln
from tqdm import tqdm
import warnings
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_logger(logger_name: str = __name__) -> logging.Logger:
    """
    Sets up a logger that logs to both console and a file located in:
    ../logs/<script_name>/<script_name>_YYYY-MM-DD_HH-MM-SS.log

    Args:
        logger_name (str): Name for the logger, default is __name__

    Returns:
        logging.Logger: Configured logger object
    """
    # Get the script file path
    script_path = Path(sys.argv[0] if '__file__' not in globals() else __file__).resolve()
    script_name = script_path.stem

    # Create logs/<script_name>/ directory relative to the script
    log_dir = script_path.parent.parent / "logs" / script_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Create and configure the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Avoid duplicate handlers if function is called multiple times

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
# Parameters
sr = 48000  # Sampling rate
peak_target_db = -1.0
loudness_target_lufs = -23.0

# Processing function
def process_audio(input_path, output_path):
    with AudioFile(input_path).resampled_to(sr) as f:
        audio = f.read(f.frames)
        rate = f.samplerate

    # Step 1: Noise Reduction
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.85)

    # Step 2: Apply Effects
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        LowpassFilter(cutoff_frequency_hz=18000),
        HighpassFilter(cutoff_frequency_hz=100),
        PeakFilter(cutoff_frequency_hz=250, gain_db=1, q=1),
        PeakFilter(cutoff_frequency_hz=500, gain_db=-1, q=1),
        PeakFilter(cutoff_frequency_hz=2000, gain_db=1.5, q=2.5),
        Compressor(threshold_db=-12, ratio=3),
        # Gain(gain_db=3),
    ])
    effected = board(reduced_noise, sr)

    # Step 3: Peak Normalization
    # peak_normalized_audio = pyln.normalize.peak(effected, peak_target_db)

    # meter = pyln.Meter(rate)
    # loudness = meter.integrated_loudness(np.permute_dims(effected))
    # loudness_normalized_audio = np.permute_dims(pyln.normalize.loudness(np.permute_dims(effected), loudness, loudness_target_lufs))
   
    # Step 4: Loudness Normalization
    meter = pyln.Meter(rate)

    # Transpose effected to shape (samples, channels) for pyloudnorm
    effected_swapped = effected.T

    # Calculate loudness
    loudness = meter.integrated_loudness(effected_swapped)

    # Normalize loudness
    normalized_audio = pyln.normalize.loudness(effected_swapped, loudness, loudness_target_lufs)

    # Transpose back to original shape (channels, samples)
    loudness_normalized_audio = normalized_audio.T
    # Save processed audio
    with AudioFile(output_path, 'w', rate, loudness_normalized_audio.shape[0]) as f:
        f.write(loudness_normalized_audio)

def process_audio_talkbank(
        logger,
        csv_path: str = "../data/labeled_audios.csv",
        output_dir: str = "../data/talkbank/processed",
        is_train: bool = True,
        use_segment: bool = False,
    ):
    csv_file = pd.read_csv(csv_path)
    csv_file = csv_file[csv_file['test_talkbank']!=is_train]
    total = csv_file.shape[0]
    for idx, row in csv_file.iterrows():
        file_name = row['segment_path' if use_segment else 'processed_path']
        unique_id = row['segment_id' if use_segment else 'unique_id']

        is_mp3 = file_name.endswith(".mp3")

        logger.info(f"📦 [{idx}/{total}] Processing {file_name}")

        new_file_name = str(unique_id)+(".mp3" if is_mp3 else ".wav")
        output_path = os.path.join(output_dir, new_file_name)
        if os.path.exists(output_path): 
            logger.info(f"⚠️ Skipping {file_name} because output already exists.")
            continue
        try:
            logger.info(f"Going to process file {file_name}")
            process_audio(file_name, output_path)
        except Exception as e:
            logger.error(f"❌ Error processing {file_name}: {e}")
        else:
            logger.info(f"✅ Processed and saved into: {output_path}")

def _process_single_file(row, output_dir, use_segment):
    file_name = row['segment_path' if use_segment else 'processed_path']
    unique_id = row['segment_id' if use_segment else 'unique_id']
    is_mp3 = file_name.endswith(".mp3")

    new_file_name = str(unique_id)+(".mp3" if is_mp3 else ".wav")
    output_path = os.path.join(output_dir, new_file_name)
    if os.path.exists(output_path): 
        return f"⚠️ Skipping {file_name} because output already exists."
    try:
        process_audio(file_name, output_path)
        return f"✅ Processed and saved into: {output_path}"
    except Exception as e:
        return f"❌ Error processing {file_name}: {e}"

def process_audio_talkbank_parallel(
        logger,
        csv_path: str = "../data/labeled_audios.csv",
        output_dir: str = "../data/talkbank/processed",
        is_train: bool = True,
        num_workers: int = os.cpu_count(),
        use_segment: bool = False,
    ):
    
    csv_file = pd.read_csv(csv_path)
    csv_file = csv_file[csv_file['test_talkbank']!=is_train]
    total = csv_file.shape[0]

    logger.info(f"🚀 Starting parallel processing with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        csv_file = csv_file.reset_index()
        for idx, row in csv_file.iterrows():
            logger.info(f"📦 [{idx}/{total}] Queued {row['segment_path' if use_segment else 'processed_path']}")
            futures.append(executor.submit(_process_single_file, row, output_dir, use_segment))

        for future in as_completed(futures):
            logger.info(future.result())

def process_audio_eureka(input_dir: str, output_dir: str): 
    # Load bad audio IDs
    with open('../data/bad_audios.txt', 'r') as file:
        bad_audio_ids = set(file.read().splitlines())

        # Suppress and handle warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        for file_name in tqdm(os.listdir(input_dir), desc="Processing", leave=True):
            if file_name.endswith(".mp3") or file_name.endswith(".wav"):
                participant_id = file_name.split('.')[0]
                if participant_id in bad_audio_ids:
                    tqdm.write(f"Skipping {file_name} due to exclusion list.")
                    continue

                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)

                try:
                    process_audio(input_path, output_path)
                except Exception as e:
                    tqdm.write(f"Error processing {file_name}: {e}")
                else:
                    tqdm.write(f"Processed and saved: {file_name}")
        
        for warning in w:
            tqdm.write(f"Warning: {warning.message}")

    print("All audio files processed!")

def main():
    parser = argparse.ArgumentParser(description="Process audio files with noise reduction, effects, and normalization.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="Input directory containing audio files (default: '../data/raw_data/train_audios/')."
    )
    parser.add_argument(
        "--csv_file", 
        type=str, 
        help="Input CSV containing audio paths and ids (default: '../data/labeled_audios.csv')."
    )
    # For test use --no-train
    parser.add_argument(
        "--train", 
        action=argparse.BooleanOptionalAction,
        help="Whether to extract train or test audios from DF."
    )
    # For processed_path and unique_id use --no-segment
    # Use --no-segment when processing full audios
    # Use --segment when processing split audios in 30s chunks
    parser.add_argument(
        "--segment", 
        action=argparse.BooleanOptionalAction,
        help="Whether to use segment_path and segment_id or processed_path and unique_id."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory to save processed audio files (default: '../data/post_data/')."
    )    
    parser.add_argument(
        "--dataset_type", 
        type=str, 
        default="eureka", 
        help="Type of dataset to process, eureka or talkbank."
    )
    logger = setup_logger()
    args = parser.parse_args()
    input_dir = args.input_dir
    csv_path = args.csv_file
    is_train = args.train
    use_segment = args.segment
    output_dir = args.output_dir
    dataset_type = args.dataset_type

    # safeguard to dataset type option
    os.makedirs(output_dir, exist_ok=True)
    if dataset_type == "eureka":
        process_audio_eureka(input_dir, output_dir)
    elif dataset_type == "talkbank-parallel": 
        process_audio_talkbank_parallel(logger, csv_path=csv_path, output_dir=output_dir, is_train=is_train, use_segment=use_segment)
    elif dataset_type == "talkbank": 
        process_audio_talkbank(logger, csv_path=csv_path, output_dir=output_dir, is_train=is_train, use_segment=use_segment)
    else:
        print(f"Not a valid datset type options: eureka talkbank")



if __name__ == "__main__":
    main()
