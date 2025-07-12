import torch
import os
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from IPython.display import Audio
from tqdm import tqdm

#TODO: CHANGE CRISPERWHISPER TO HOW IT IS DONE WITH LLAMA

def translate_with_llama(text_input, tokenizer, model, device, source_lang):
    # Craft a prompt for translation
    prompt = f"Translate COMPLETELY the folling text from {source_lang} to English, the STOP generation: \n{text_input}\nTranslation:"
    
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
    
    return translated_text

def transcribe_audios():
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

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device
    )

    result = transcriber(file_path)
    return result

def translate_audios():
    # Load the dataset
    metadata_file = pd.read_csv('non_english_transcriptions_CrisperWhisperLarge.csv')
    metadata_file['translation'] = ''
    
    # Define language mappings
    languages_dict = {'spanish': 'Spanish', 'mandarin': 'Mandarin', 'german': 'German'}
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the LLaMA model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with the actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  device_map="auto",
                                                  torch_dtype=torch.float16)
    
    # Iterate over each language
    for language_key, language_name in languages_dict.items():
        # Filter rows for the current language
        metadata_file_lgn = metadata_file[metadata_file['language'] == language_key]
        
        for idx, row in tqdm(metadata_file_lgn.iterrows(), total=len(metadata_file_lgn)):
            transcribed_text = row['transcribed_text']
            translated_text = translate_with_llama(transcribed_text, tokenizer, model, device, language_name)
            metadata_file.at[idx, 'translation'] = translated_text
    
    # Save the translations to a new CSV file
    metadata_file.to_csv('non_english_translations_CrisperWhisperLarge_LLaMA_3_3B_NewPrompt.csv', index=False)


def transcribe(file_path):
    # Load the processor and model
    processor = WhisperProcessor.from_pretrained("nyrahealth/CrisperWhisper")
    model = WhisperForConditionalGeneration.from_pretrained("nyrahealth/CrisperWhisper")

    # Set the desired language and task
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="es", task="transcribe")  # 'es' for Spanish

    # Prepare your audio input (ensure it's a NumPy array or a list of NumPy arrays)
    inputs = processor(audio_input, return_tensors="pt")

    # Generate transcription with specified language
    predicted_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription[0])


def transcribe_audios(file_path):
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

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device
    )

    result = transcriber(file_path)
    return result

if __name__ == "__main__":
    #audio_list = sorted(os.listdir("/media/data/shared/eureka/phase3/talkbank_dementia_processed/Spanish/Ivanova/AD/"))
    #audio_list.remove('AD-W-89-20.mp3')
    audio_list = ['AD-M-82-321.mp3']
    # audio_path = "/media/data/shared/eureka/phase3/talkbank_dementia_processed/Spanish/Ivanova/AD/AD-M-78-24.mp3"  # Replace with your audio file path
    for x in audio_list:
        audio_path = os.path.join("/media/data/shared/eureka/phase3/talkbank_dementia_processed/Spanish/Ivanova/AD/", x)
        print(f"Processing {audio_path}...")
        transcription = transcribe_audio(audio_path)
        print("Transcription:")
        print(transcription["text"])
