import torch
import os
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from IPython.display import Audio
from tqdm import tqdm
import argparse

def translate_with_llama(text_input, tokenizer, model, device, source_lang):
    # Craft a prompt for translation
    prompt = f"Translate COMPLETELY the folling text from {source_lang} to English, then STOP the generation: \n{text_input}\nTranslation:"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the translation
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode the generated tokens
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the translation from the output
    translation = translated_text.split("Translation:")[1].strip()
    
    return translation

def transcribe_audios(audios_path, intermediate_files_path):
    info_file = pd.read_csv(os.path.join(intermediate_files_path, 'all_audios_original_labels.csv'))

    languages_ids = {'spanish':'es',
                     'german':'de',
                     'mandarin':'zh'}

    all_audios = os.listdir(audios_path)
    unique_ids = [int(x.replace('.mp3','')) for x in all_audios]

    info_file = info_file[info_file['unique_id'].isin(unique_ids)]
    info_file = info_file[info_file['language']!='english']

    info_file.reset_index(drop=True, inplace=True)

    info_file['transcription'] = ""
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    for idx, participant_info in tqdm(info_file.iterrows(), total=len(info_file)):

        file_path = os.path.join(audios_path, str(participant_info['unique_id'])+'.mp3')
        
        language = participant_info['language']
        language_ref = languages_ids[language]

        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps="word",
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"language": language_ref},
            ignore_warning=True
            )
        try:
            result = transcriber(file_path)['text']
        except:
            continue

        info_file.loc[idx, 'transcription'] = result
    
    info_file.to_csv(os.path.join(intermediate_files_path,'nonenglish_audios_with_transcriptions.csv'), index=False)

    return info_file

def translate_audios(info_dataframe, intermediate_files_path):
    # Load the dataset
    
    info_dataframe['translation'] = ''
    
    # Define language mappings
    languages_dict = {'spanish': 'Spanish', 'mandarin': 'Mandarin', 'german': 'German'}
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the LLaMA model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  device_map="auto",
                                                  torch_dtype=torch.float16)
    
    # Iterate over each language
    for language_key, language_name in languages_dict.items():
        # Filter rows for the current language
        metadata_file_lgn = info_dataframe[info_dataframe['language'] == language_key]
        
        for idx, row in tqdm(metadata_file_lgn.iterrows(), total=len(metadata_file_lgn)):

            transcribed_text = row['transcription']
            translated_text = translate_with_llama(transcribed_text, tokenizer, model, device, language_name)
            info_dataframe.at[idx, 'translation'] = translated_text
    
    # Save the translations to a new CSV file
    info_dataframe.to_csv(os.path.join(intermediate_files_path,'nonenglish_audios_with_transcriptions_and_translations.csv'), index=False)

def main(audios_path, intermediate_files_path):

    info_dataframe = transcribe_audios(audios_path, intermediate_files_path)

    translate_audios(info_dataframe, intermediate_files_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process TalkBank dementia data.")
    parser.add_argument('--intermediate_files_path', type=str, default='../../data/interim', help="Path to the intermediate files")
    parser.add_argument('--audios_path', type=str, default='../../data/processed/post_data/test_audios', help="Path to the files to transcribe and translate")
    args = parser.parse_args()
    main(args.audios_path, args.intermediate_files_path) 

    
