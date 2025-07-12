import torch
import os
import pandas as pd
from tqdm import tqdm
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import nltk
nltk.download('averaged_perceptron_tagger_eng')

#TODO: ARGS FOR paths and files. 

def main():
    
    metadata_file = pd.read_csv('/media/data/home/pruiz/Eureka-Phase3/translate_audios/non_english_translations_CrisperWhisperLarge_LLaMA_3_8B_postprocessed.csv')
    output_dir = '/buckets/projects/eureka/data_exp1/post_data/openvoice_translation_llama_V1/'

    #OPENVOICE
    print("Loading OpenVoice...")
    ckpt_converter = '/media/data/home/pruiz/Eureka-Phase3/translate_audios/OpenVoice/checkpoints_v2/converter'
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    speed = 1.0
    model_op = TTS(language='EN_NEWEST', device=device)
    metadata_file = metadata_file[(metadata_file['translation'] != " ")]
    breakpoint()

    for idx, row in tqdm(metadata_file.iterrows()):

        src_path = f'{output_dir}/tmp.wav'

        print(f"Processing {row['unique_id']} ...")
        audio_path = os.path.join('/buckets/projects/eureka/data_exp1/post_data/train_audios',f"{row['unique_id']}.mp3")
        if not os.path.exists(audio_path):
            audio_path = os.path.join('/buckets/projects/eureka/data_exp1/post_data/test_audios',f"{row['unique_id']}.mp3")
        
        print(audio_path)
        print("Loading audio & transcription...")
        target_se, audio_name = se_extractor.get_se(audio_path, tone_color_converter, vad=True)

        print("Creating audio")
        speaker_ids = model_op.hps.data.spk2id
    
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            
            source_se = torch.load(f'/media/data/home/pruiz/Eureka-Phase3/translate_audios/OpenVoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
            if torch.backends.mps.is_available() and device == 'cpu':
                torch.backends.mps.is_available = lambda: False
            model_op.tts_to_file(row['translation'], speaker_id, src_path, speed=speed)
            save_path = f'{output_dir}/{row["unique_id"]}_output_v2_{speaker_key}.wav'

            # Run the tone color converter
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=save_path,
                message=encode_message)
        
            

if __name__ == "__main__":
    main()