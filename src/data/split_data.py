import os
import pandas as pd
import random
from moviepy import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

def get_diagnosis_transcript(path):
    with open(path) as file:
        lines = [line for line in file.readlines() if line.startswith('@ID') and 'PAR' in line]
        return lines[0].split('|')[5]

def process_names_hopkins(df,row):
    name = row['File']
    name = name.replace('(no audio file)','').replace('(see note on next tab)','').replace('(no audio recording)','').strip()
    row['File'] =  name
    if pd.notna(row['Type']):
        return row
    if not name[-1].isdigit():
        name = name[:-1]
    filtered = df[(df['File'].str.contains(name))&(df['Type'].notna())]
    if len(filtered) > 0:
        row['Type'] = filtered.iloc[0]['Type']
    return row

def ivanova(base_path, df):
    for diagnosis in ['HC','MCI','AD']:
        path = os.path.join(base_path,'Spanish/Ivanova',diagnosis)
        for file in os.listdir(path):
            if file.endswith('.mp3'):
                df.loc[len(df)] = ['spanish','ivanova',file.replace('.mp3',''),os.path.join(path,file),diagnosis]
    
    return df

def vas(base_path, df):
    labels = pd.read_excel('./metadata_files/VAS_labels.xlsx')
    path = os.path.join(base_path,'English/VAS')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id = int(file.replace('.mp3',''))
            diagnosis = labels[labels['VAS ID']==file_id].iloc[0]['H/MCI/D']
            df.loc[len(df)] = ['english','vas',file_id,os.path.join(path,file),diagnosis]   

    return df

def depaul(base_path, df):
    path = os.path.join(base_path,'English/DePaul')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            df.loc[len(df)] = ['english','depaul',1,os.path.join(path,file),'semantic PPA']
    return df

def holland(base_path, df):
    path = os.path.join(base_path+"_processed",'English/Holland')
    for file in os.listdir(path):
        if file.endswith('.mp3') and 'participant' in file:
            df.loc[len(df)] = ['english','holland',file.replace('.mp3',''),os.path.join(path,file),'other_dem']
    
    return df

def hopkins(base_path, df):
    labels = pd.read_excel('./metadata_files/Hopkins-meta.xlsx')
    labels = labels.apply(lambda x: process_names_hopkins(labels,x),axis=1)
    labels = labels[labels['Type'].notna()]

    path = os.path.join(base_path,'English/Hopkins')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id = file.replace('.mp3','')
            if len(labels[labels['File']==file_id]) == 0:
                continue
            diagnosis = labels[labels['File']==file_id].iloc[0]['Type']
            df.loc[len(df)] = ['english','hopkins',file_id,os.path.join(path,file),diagnosis]

    return df   

def kempler(base_path, df):
    path = os.path.join(base_path,'English/Kempler')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id = file.replace('.mp3','')
            df.loc[len(df)] = ['english','kempler',file_id,os.path.join(path,file),'AD']   
    return df

def lanzi(base_path, df):

    with VideoFileClip(os.path.join(base_path,'English/Lanzi/Treatment/11-06-17.mp4')) as video:
        audio = video.audio
        audio.write_audiofile(os.path.join(base_path,'English/Lanzi/Treatment/11-06-17.mp3'))

    with VideoFileClip(os.path.join(base_path,'English/Lanzi/Treatment/11-14-17.mp4')) as video:
        audio = video.audio
        audio.write_audiofile(os.path.join(base_path,'English/Lanzi/Treatment/11-14-17.mp3'))

    path = os.path.join(base_path,'English/Lanzi')
    for folder in os.listdir(path):
        sub_path = os.path.join(path,folder)
        for file in os.listdir(sub_path):
            if file.endswith('.mp3'):
                file_id = file.replace('.mp3','')
                df.loc[len(df)] = ['english','lanzi',file_id,os.path.join(sub_path,file),'mNCD']

    return df

def lu_english(base_path, base_path_transcripts, df):
        
    path = os.path.join(base_path,'English/Lu')
    transcript_path = os.path.join(base_path_transcripts,'English/Lu')
    for folder in os.listdir(path):
        sub_path = os.path.join(path,folder)
        for file in os.listdir(sub_path):
            if file.endswith('.mp3'):
                file_id = file.replace('.mp3','')
                diagnosis = get_diagnosis_transcript(os.path.join(transcript_path,folder,file_id+'.cha'))
                df.loc[len(df)] = ['english','lu',file_id,os.path.join(sub_path,file),diagnosis]

    return df

def pitt(base_path, base_path_transcripts, df):
    for group in ['Control','Dementia']:
        path = os.path.join(base_path,'English/Pitt',group)
        transcript_path = os.path.join(base_path_transcripts,'English/Pitt',group)
        for folder in os.listdir(path):
            sub_path = os.path.join(path,folder)
            for file in os.listdir(sub_path):
                if file.endswith('.mp3'):
                    file_id = file.replace('.mp3','')
                    diagnosis = get_diagnosis_transcript(os.path.join(transcript_path,folder,file_id+'.cha'))
                    df.loc[len(df)] = ['english','pitt',file_id,os.path.join(sub_path,file),diagnosis]

    return df

def baycrest(base_path,base_path_transcripts, df):
    path = os.path.join(base_path,'English/Protocol/Baycrest')
    transcript_path = os.path.join(base_path_transcripts,'English/Protocol/Baycrest')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id = file.replace('.mp3','')
            diagnosis = get_diagnosis_transcript(os.path.join(transcript_path,file_id+'.cha'))
            df.loc[len(df)] = ['english','baycrest',file_id,os.path.join(path,file),diagnosis]

    return df

def delaware(base_path,base_path_transcripts,df):
    path = os.path.join(base_path,'English/Protocol/Delaware')
    transcript_path = os.path.join(base_path_transcripts,'English/Protocol/Delaware')
    for folder in os.listdir(path):
        sub_path = os.path.join(path,folder)
        for file in os.listdir(sub_path):
            if file.endswith('.mp3'):
                file_id = file.replace('.mp3','')
                diagnosis = get_diagnosis_transcript(os.path.join(transcript_path,folder,file_id+'.cha'))
                df.loc[len(df)] = ['english','delaware',file_id,os.path.join(sub_path,file),diagnosis]

    return df

def lu_mandarin(base_path, df):
    path = os.path.join(base_path,'Mandarin/Lu')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id =  file.replace('.mp3','')
            df.loc[len(df)] = ['mandarin','lu',int(file_id),os.path.join(path,file),'dementia']
    
    return df

def wls(base_path, df):
    path = os.path.join(base_path,'English/WLS')
    for folder in os.listdir(path):
        if folder == '0extra':
            continue
        sub_path = os.path.join(path,folder)
        for file in os.listdir(sub_path):
            if file.endswith('.mp3'):
                file_id = file.replace('.mp3','')
                df.loc[len(df)] = ['english','wls',file_id,os.path.join(sub_path,file),'Control']

def chou(base_path, df):
    for diagnosis in ['HC','MCI']:
        path = os.path.join(base_path,'Mandarin/Chou',diagnosis)
        for folder in os.listdir(path):
            sub_path = os.path.join(path,folder)
            for file in os.listdir(sub_path):
                if file.endswith('.mp3'):
                    df.loc[len(df)] = ['mandarin','chou',file.replace('.mp3',''),os.path.join(sub_path,file),diagnosis]

def ye(base_path, df):
    path = os.path.join(base_path,'Mandarin/Ye')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id =  file.replace('.mp3','')
            df.loc[len(df)] = ['mandarin','ye',file_id,os.path.join(path,file),'mci']

    return df

def jalvingh(base_path, df):
    path = os.path.join(base_path,'German/Jalvingh')
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            file_id =  file.replace('.mp3','')
            if 'FTD' in file_id:
                diagnosis = 'ftd'
            elif 'Dement' in file_id:
                diagnosis = 'dementia'
            elif 'MCI' in file_id:
                diagnosis = 'mci'
            elif 'PPA' in file_id:
                diagnosis = file_id.split('_')[1].lower()
            df.loc[len(df)] = ['german','jalvingh',file_id,os.path.join(path,file),diagnosis]
    
    return df

def map_diagnosis(value):
    map_diagnosis_dict = {
    'hc': ['control','hc','h','conrol'],
    'ad': ['ad','alzheimer\'s','probablead','possiblead','probable'],
    'other_dem': ['dementia','d','vascular','ppa-nos','lvppa','svppa','nfappa','ftd','other','semantic ppa','pick\'s','aphasia','sd','lpa','pnfa'],
    'mci': ['mci','memory','mncd']
    }

    for i in map_diagnosis_dict:
        if value in map_diagnosis_dict[i]:
            return i
    return value

def map_uid_phase2(base_path, df):

    challenge_map = pd.read_csv('./metadata_files/dementiabank_uid_map.csv')
    test_audios = pd.read_csv('./metadata_files/acoustic_test_labels.csv')

    challenge_map['test'] = challenge_map.apply(lambda x: x['uid'] in test_audios['uid'].values,axis=1)
    challenge_map['train'] = challenge_map.apply(lambda x: x['uid'] not in test_audios['uid'].values,axis=1)

    df = df.merge(challenge_map[['corpus', 'dementiabank_id', 'train', 'test']], left_on=['study', 'id'], right_on=['corpus', 'dementiabank_id'], how='left')
    df = df.drop(columns=['corpus', 'dementiabank_id'])
    df = df.rename(columns={'train':'train_eureka','test':'test_eureka'})

    return df

def get_processed_path(row):

    file_path = row['file_path']
    if 'talkbank_dementia_processed' in file_path:
        return file_path
    else:
        return file_path.replace('talkbank_dementia','talkbank_dementia_processed')

def standardize_labels(base_path,df,saving_path):

    df['diagnosis'] = df['diagnosis'].str.lower()
    df = df[df['diagnosis']!='']
    df.loc[:,'diagnosis'] = df['diagnosis'].apply(map_diagnosis)

    df = map_uid_phase2(base_path, df)

    df['processed_path'] = df.apply(get_processed_path,axis=1)
    #TODO: MISSING TO SKIP TWO AUDIOS:
    """
    Audios to skip:
    English/WLS/13/13958 (empty audio after segmentation): 946112
    English/WLS/09/09968 (too short): 921874
    """

    random.seed(42)
    codes = random.sample(range(100000, 999999), len(df))
    df['unique_id'] = codes

    df.to_csv('files/all_audios_original_labels.csv',index=False)

    return df

def split_audios(base_path, df):
    # Parameters
    chunk_duration_ms = 30 * 1000  # 30 seconds in milliseconds
    output_dir = base_path+'_split'
    os.makedirs(output_dir, exist_ok=True)
    # Placeholder for new rows
    new_rows = []

    print("Total df size:",len(df))
    for _, row in tqdm(df.iterrows()):
        uid = row['unique_id']
        path = row['processed_path']

        if os.path.exists(os.path.join(output_dir,f"{uid}_0.mp3")) or path=='':
            continue
        try:
            audio = AudioSegment.from_file(path)
            duration_ms = len(audio)
            num_chunks = (duration_ms + chunk_duration_ms - 1) // chunk_duration_ms  # ceil division

            for i in range(num_chunks):
                new_uid = f"{uid}_{i}"
                out_path = os.path.join(output_dir, f"{new_uid}.mp3")

                start_ms = i * chunk_duration_ms
                end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
                chunk = audio[start_ms:end_ms]
                chunk.export(out_path, format="mp3")

                new_rows.append(row.to_dict()|{'segment_id': new_uid, 'segment_path': out_path})

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Create new DataFrame
    df = pd.DataFrame(new_rows)

    df.to_csv('files/split_audios_original_labels.csv', index=False)

    return df

def check_minimum_length(df):

    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _check_audio_length(path):
        file = AudioSegment.from_file(path)
        if file.duration_seconds < 0.4:
            return path

    short_audios = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for idx, row in df.iterrows():
            futures.append(executor.submit(_check_audio_length, row['segment_path']))
        for future in as_completed(futures):
            if future.result() is not None:
                short_audios.append(future.result())

    df = df[~df['segment_path'].isin(short_audios)]
    
    return df

def data_split_segments(df):

    df.loc[:,'test_talkbank'] = df.loc[:,'test_eureka'].apply(lambda x: x if not pd.isna(x) else False)
    df.loc[df['language']=='mandarin','test_talkbank'] = True
    df.loc[df['study']=='vas','test_talkbank'] = True
    df.loc[df['study']=='holland','test_talkbank'] = True

    #TODO: EXPLAIN WHY WE ARE DOING THIS
    include_test = {'study':['delaware','wls','wls','wls','wls','wls','wls','wls','jalvingh','jalvingh','jalvingh','jalvingh'], 
                    'id':['04-1','14895','11928','09636','10723','04532','15842','02213','PPA_LPA','Park_MCI04','FTD03','FTD01'],}    
    df.loc[df['study'].isin(include_test['study']) & df['id'].isin(include_test['id']), 'test_talkbank'] = True

    df.to_csv('files/split_audios_final_partition.csv',index=False)

    return df

def generate_files_training_segments(df):
    #METADATA
    df_metadata = df[['segment_id','test_talkbank']].copy()
    df_metadata['uid'] = df_metadata['segment_id']

    df_metadata['split'] = df_metadata.apply(lambda x: "train" if not x['test_talkbank'] else "test",axis=1)

    df_metadata = df_metadata.drop(columns=['segment_id','test_talkbank'])
    df_metadata.to_csv('files/Metadata.csv',index=False)

    #ALL LABELS
    df_label = df[['segment_id','diagnosis']].copy()
    df_label['uid'] = df_label['segment_id']

    df_label['diagnosis_control'] = df_label.apply(lambda x: 1 if x['diagnosis']=='hc' else 0,axis=1)
    df_label['diagnosis_mci'] = df_label.apply(lambda x: 1 if x['diagnosis']=='mci' else 0,axis=1)
    df_label['diagnosis_adrd'] = df_label.apply(lambda x: 1 if x['diagnosis']=='ad' else 0,axis=1)
    df_label['diagnosis_other'] = df_label.apply(lambda x: 1 if x['diagnosis']=='other_dem' else 0,axis=1)

    df_label = df_label.drop(columns=['diagnosis','segment_id'])
    df_label.to_csv('files/All_labels.csv',index=False)

    #TRAINING LABELS
    df_label = df[df['test_talkbank']==False][['segment_id','diagnosis']].copy()
    df_label['uid'] = df_label['segment_id']

    df_label['diagnosis_control'] = df_label.apply(lambda x: 1 if x['diagnosis']=='hc' else 0,axis=1)
    df_label['diagnosis_mci'] = df_label.apply(lambda x: 1 if x['diagnosis']=='mci' else 0,axis=1)
    df_label['diagnosis_adrd'] = df_label.apply(lambda x: 1 if x['diagnosis']=='ad' else 0,axis=1)
    df_label['diagnosis_other'] = df_label.apply(lambda x: 1 if x['diagnosis']=='other_dem' else 0,axis=1)

    df_label = df_label.drop(columns=['diagnosis','segment_id'])
    df_label.to_csv('files/Train_labels.csv',index=False)

    #TEST LABELS
    df_label = df[df['test_talkbank']==True][['segment_id','diagnosis']].copy()
    df_label['uid'] = df_label['segment_id']

    df_label['diagnosis_control'] = df_label.apply(lambda x: 1 if x['diagnosis']=='hc' else 0,axis=1)
    df_label['diagnosis_mci'] = df_label.apply(lambda x: 1 if x['diagnosis']=='mci' else 0,axis=1)
    df_label['diagnosis_adrd'] = df_label.apply(lambda x: 1 if x['diagnosis']=='ad' else 0,axis=1)
    df_label['diagnosis_other'] = df_label.apply(lambda x: 1 if x['diagnosis']=='other_dem' else 0,axis=1)

    df_label = df_label.drop(columns=['diagnosis','segment_id'])
    df_label.to_csv('files/Test_labels.csv',index=False)  

    #Eureka TEST
    df_label = df[(df['processed_path']!='')&(df['test_eureka']==True)][['segment_id','diagnosis']].copy()
    df_label['uid']=df_label['segment_id']

    df_label['diagnosis_control'] = df_label.apply(lambda x: 1 if x['diagnosis']=='hc' else 0,axis=1)
    df_label['diagnosis_mci'] = df_label.apply(lambda x: 1 if x['diagnosis']=='mci' else 0,axis=1)
    df_label['diagnosis_adrd'] = df_label.apply(lambda x: 1 if x['diagnosis']=='ad' or x['diagnosis']=='other_dem' else 0,axis=1)

    df_label = df_label.drop(columns=['diagnosis','segment_id'])
    df_label.to_csv('files/Test_labels_eureka.csv',index=False)

    df[(df['study']=='vas')|(df['language']=='mandarin')][['segment_id','study','language']].to_csv('./files/out_of_distribution_data.csv',index=False)


    return df_metadata, df_label


def main():

    base_path = '/media/data/shared/ct_igc_phase3/phase3/talkbank_dementia'
    base_path_transcripts = '/media/data/shared/eureka/ct_igc_phase3/talkbank_dementia_transcripts'
    df = pd.DataFrame({'language':[],'study':[],'id':[],'file_path':[],'diagnosis':[]})

    #Ivanova
    df = ivanova(base_path, df)
    
    #VAS
    df = vas(base_path, df)

    #DePaul
    df = depaul(base_path, df)

    #Holland
    df = holland(base_path, df)

    #Hopkins
    df = hopkins(base_path, df)

    #Kempler
    df = kempler(base_path, df)

    #Lanzi
    df = lanzi(base_path, df)

    #Lu - English
    df = lu_english(base_path, base_path_transcripts, df)

    #Pitt
    df = pitt(base_path, base_path_transcripts, df)

    #Baycrest
    df = baycrest(base_path, base_path_transcripts, df)

    #Delaware
    df = delaware(base_path, base_path_transcripts, df)

    #WLS
    df = wls(base_path, df)

    #Chou
    df = chou(base_path, df)

    #Lu - Mandarin
    df = lu_mandarin(base_path, df)

    #Ye
    df = ye(base_path, df)

    #Jalvingh
    df = jalvingh(base_path, df)

    df = standardize_labels(base_path,df)

    df = split_audios(base_path, df)

    df = check_minimum_length(df)

    df = data_split_segments(df)

    df_metadata, df_label = generate_files_training_segments(df)

    #TODO: No entiendo bien la diferencia entre la sección
    # get splits for full audios (no segments) 





if __name__ == "__main__":
    main()
    #TODO: ADD ARGS.base_path TO THE BASE PATH
    #TODO: ADD ARGS.files_saving_path TO SAVE THE FILES -> Some to the interim folder some to the processed folder
    #TODO: CHANGE ALL .to_csv() to use the ARGS.files_saving_path

