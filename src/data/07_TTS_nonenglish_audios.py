import torch
import os
import pandas as pd
from tqdm import tqdm
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import nltk
nltk.download('averaged_perceptron_tagger_eng')
import argparse


def main(intermediate_files_path,audios_path, output_path, path_checkpoints):
    
    metadata_file = pd.read_csv(os.path.join(intermediate_files_path, 'nonenglish_audios_with_transcriptions_and_translations.csv'))
    output_dir = output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    #OPENVOICE
    print("Loading OpenVoice...")
    ckpt_converter = os.path.join(path_checkpoints, 'converter')
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    speed = 1.0
    model_op = TTS(language='EN_NEWEST', device=device)
    metadata_file = metadata_file[(metadata_file['translation'] != " ")]
    
    breakpoint()

    for idx, row in tqdm(metadata_file.iterrows()):

        if idx > 10:
            continue

        src_path = f'{output_dir}/tmp.wav'

        print(f"Processing {row['unique_id']} ...")
        audio_path = os.path.join(audio_path, f"{row['unique_id']}.mp3")
        
        print(audio_path)
        print("Loading audio & transcription...")
        target_se, audio_name = se_extractor.get_se(audio_path, tone_color_converter, vad=True)

        print("Creating audio")
        speaker_ids = model_op.hps.data.spk2id
    
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            
            source_se = torch.load(os.path.join(path_checkpoints,f'base_speakers/ses/{speaker_key}.pth'), map_location=device)
            if torch.backends.mps.is_available() and device == 'cpu':
                torch.backends.mps.is_available = lambda: False
            model_op.tts_to_file(row['translation'], speaker_id, src_path, speed=speed)
            save_path = os.path.join(output_dir,f'{row["unique_id"]}_output_v2_{speaker_key}.wav')

            # Run the tone color converter
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=save_path,
                message=encode_message)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process TalkBank dementia data.")
    parser.add_argument('--intermediate_files_path', type=str, default='../../data/interim', help="Path to the intermediate files")
    parser.add_argument('--audios_path', type=str, default='../../data/processed/post_data/test_audios', help="Path to the files to transcribe and translate")
    parser.add_argument('--output_path', type=str, default='../../data/processed/post_data/translated_audios', help="Path to write the translated audios")
    parser.add_argument('--path_checkpoints', type=str, default='', help='Path to OpenVoice checkpoints')
    args = parser.parse_args()

    main(args.intermediate_files_path, args.audios_path, args.output_path, args.path_checkpoints)