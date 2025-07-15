import os
import pandas as pd
import glob
import argparse

def get_metadata_lu_eng(path):
    with open(path) as file:
        lines = [line for line in file.readlines() if line.startswith('@ID') and 'PAR' in line]
        age, gender = lines[0].split('|')[3].replace(';',''), lines[0].split('|')[4]
        return age, gender 
    
def get_metadata(path):
    with open(path) as file:
        lines = [line for line in file.readlines() if line.startswith('@ID') and 'PAR' in line]
        age, gender = lines[0].split('|')[3][0:2], lines[0].split('|')[4]
        return age, gender 

def main(metadata_path, file_saving_path, transcripts_path, training_files):

    labels_data = pd.read_csv(os.path.join(file_saving_path, 'all_audios_original_labels.csv'))

    labels_data['gender'] = ""
    labels_data['age'] = ""
    labels_data['education'] = ""

    #ivanova
    print("Processing Ivanova")
    ivanova = pd.read_excel(os.path.join(metadata_path,'Ivanova-meta.xlsx'))
    for idx, row in ivanova.iterrows():
        id = row['Identifier']
        gender = row['Gender']
        age = row['Age']
        education = row['Schooling years']
        labels_data.loc[(labels_data['study'] == 'ivanova') & (labels_data['id'] == id),'gender'] = gender
        labels_data.loc[(labels_data['study'] == 'ivanova') & (labels_data['id'] == id),'age'] = age
        labels_data.loc[(labels_data['study'] == 'ivanova') & (labels_data['id'] == id),'education'] = education

    #vas
    print("Processing VAS")

    vas = pd.read_excel(os.path.join(transcripts_path, 'English','VAS','0demo.xlsx'))
    for idx, row in vas.iterrows():
        try:
            id = str(int(row['VAS ID']))
        except:
            continue
        gender = row['gender']
        age = row['age']
        education = ""
        labels_data.loc[(labels_data['study'] == 'vas') & (labels_data['id'] == id),'gender'] = gender
        labels_data.loc[(labels_data['study'] == 'vas') & (labels_data['id'] == id),'age'] = age
        labels_data.loc[(labels_data['study'] == 'vas') & (labels_data['id'] == id),'education'] = education

    #DePaul
    print("Processing DePaul")

    labels_data.loc[labels_data["study"]== "depaul", 'gender'] = 'female'
    labels_data.loc[labels_data["study"]== "depaul", 'age'] = 66

    #Holland
    print("Processing Holland")

    #Participant_01: he's sixty two years of age &-um and has had fifteen years of education .
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_1'), 'gender'] = 'male'
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_1'), 'age'] = 62
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_1'), 'education'] = 15

    #Participant_02: is seventy six years old she received a Master's degree after eighteen years of education and was a high school history teacher .
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_2'), 'gender'] = 'female'
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_2'), 'age'] = 76
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_2'), 'education'] = 18

    #Participant_03: she is seventy eight years old and has had sixteen years of education
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_3'), 'gender'] = 'female'
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_3'), 'age'] = 80
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_3'), 'education'] = 16

    #Participant_04: No INFO ON HIM
    labels_data.loc[(labels_data['study'] == 'holland') & (labels_data['id'] == 'participant_4'), 'gender'] = 'male'

    #Hopkins
    print("Processing Hopkins")

    hopkins = pd.read_excel(os.path.join(metadata_path,'Hopkins-meta.xlsx'))
    for idx, row in hopkins.iterrows():
        id = row['File']
        gender = row['Sex']
        age = row['Age']
        education = row['Education']
        labels_data.loc[(labels_data['study'] == 'hopkins') & (labels_data['id'] == id),'gender'] = gender
        labels_data.loc[(labels_data['study'] == 'hopkins') & (labels_data['id'] == id),'age'] = age
        labels_data.loc[(labels_data['study'] == 'hopkins') & (labels_data['id'] == id),'education'] = education

    #Kempler
    print("Processing Kempler")

    participant_info = {'d1':[74, 11, 'M'],
                    'd4':[82, 'BA', 'F'],
                    'd5':['','','M'],
                    'd6':[87, '', 'F'],
                    'd6cookie':[87, '', 'F'],
                    'd9':[65, 11, 'M'],
                    'd10':[82, '8th grade', 'M']}
    for id, info in participant_info.items():
        labels_data.loc[(labels_data['study'] == 'kempler') & (labels_data['id'] == id), 'age'] = info[0]
        labels_data.loc[(labels_data['study'] == 'kempler') & (labels_data['id'] == id), 'education'] = info[1]
        labels_data.loc[(labels_data['study'] == 'kempler') & (labels_data['id'] == id), 'gender'] = info[2]

    # Lanzi
    print("Processing Lanzi")

    # Info not available in all of the samples
    dict_participants = {'538':[75,'','female'], #Africa
                     '539': ['','','female'],
                     '542': ['','','female'],
                     '541': [77,'','female'], #Caucasian
                     '544': [78,'','female'],
                     '545':['','','female'],
                     '11-06-17':['','','female'],
                     '11-14-17':['','','female']}
 
    for id, info in dict_participants.items():
        labels_data.loc[(labels_data['study'] == 'lanzi') & (labels_data['id'] == id), 'age'] = info[0]
        labels_data.loc[(labels_data['study'] == 'lanzi') & (labels_data['id'] == id), 'education'] = info[1]
        labels_data.loc[(labels_data['study'] == 'lanzi') & (labels_data['id'] == id), 'gender'] = info[2]

    #Lu
    print("Processing Lu-English")

    path_files_c = os.listdir(os.path.join(transcripts_path,'English/Lu/Control'))
    path_files_c.remove('.files')
    path_files_c = ['Control/' + file for file in path_files_c]
    path_files_d = os.listdir(os.path.join(transcripts_path,'English/Lu/Dementia'))
    path_files_d.remove('.files')
    path_files_d = ['Dementia/' + file for file in path_files_d]
    path_files_c.extend(path_files_d)

    for file in path_files_c:
        id = file.split('/')[1].replace('.cha','')
        path_file = os.path.join(os.path.join(transcripts_path,'English/Lu', file))
        age, gender = get_metadata_lu_eng(path_file)
        labels_data.loc[(labels_data['study'] == 'lu') & (labels_data['id'] == id), 'age'] = age
        labels_data.loc[(labels_data['study'] == 'lu') & (labels_data['id'] == id), 'gender'] = gender

    #Pitt
    print("Processing Pitt")

    path_files_c = glob.glob(os.path.join(transcripts_path,'English/Pitt/*/*/*.cha'))
    for file in path_files_c:
        id = file.split('/')[-1].replace('.cha','')
        age, gender = get_metadata(file)
        if age != '':
            labels_data.loc[(labels_data['study'] == 'pitt') & (labels_data['id'] == id), 'age'] = age
        labels_data.loc[(labels_data['study'] == 'pitt') & (labels_data['id'] == id), 'gender'] = gender

    #Baycrest

    print("Processing Baycrest")

    path_files_c = glob.glob(os.path.join(transcripts_path,'English/Protocol/Baycrest/*.cha'))
    for file in path_files_c:
        id = file.split('/')[-1].replace('.cha','')
        age, gender = get_metadata(file)
        if age != '':
            labels_data.loc[(labels_data['study'] == 'baycrest') & (labels_data['id'] == id), 'age'] = age
        labels_data.loc[(labels_data['study'] == 'baycrest') & (labels_data['id'] == id), 'gender'] = gender

    #Delaware
    print("Processing Delaware")


    path_files_c = glob.glob(os.path.join(transcripts_path,'English/Protocol/Delaware/*/*.cha'))
    for file in path_files_c:
        id = file.split('/')[-1].replace('.cha','')
        age, gender = get_metadata(file)
        if age != '':
            labels_data.loc[(labels_data['study'] == 'delaware') & (labels_data['id'] == id), 'age'] = age
        labels_data.loc[(labels_data['study'] == 'delaware') & (labels_data['id'] == id), 'gender'] = gender

    #WLS
    print("Processing WLS")

    wls = pd.read_excel(os.path.join(metadata_path, 'WLS-data.xlsx'))
    sex_mapping = {'':'',1:'male',2:'female'}
    for idx, row in wls.iterrows():
        id = str(row['idtlkbnk'])[-7:-2]
        gender = sex_mapping[row['sex']]
        age = row['age 2011']
        education = row['education']
        labels_data.loc[(labels_data['study'] == 'wls') & (labels_data['id'] == id),'gender'] = gender
        labels_data.loc[(labels_data['study'] == 'wls') & (labels_data['id'] == id),'age'] = age
        labels_data.loc[(labels_data['study'] == 'wls') & (labels_data['id'] == id),'education'] = education

    #Chou
    print("Processing Chou")


    chou = pd.read_excel(os.path.join(transcripts_path, 'Mandarin','Chou','0docs','0demo.xlsx'), sheet_name='df')
    diagnosis = {'M': 'mci',
                 'N': 'hc'}
    for idx, row in chou.iterrows():
        diag = diagnosis[row['CTH_ID'][0]]
        id = row['CTH_ID'][-3:]
        gender = row['Gender']
        age = row['Age']
        education = row['edu']
        labels_data.loc[(labels_data['study'] == 'chou') & (labels_data['id'].str.split('_').str[0] == id) & (labels_data['diagnosis'] == diag),'gender'] = gender
        labels_data.loc[(labels_data['study'] == 'chou') & (labels_data['id'].str.split('_').str[0] == id) & (labels_data['diagnosis'] == diag),'age'] = age
        labels_data.loc[(labels_data['study'] == 'chou') & (labels_data['id'].str.split('_').str[0] == id) & (labels_data['diagnosis'] == diag),'education'] = education

    #Lu
    print("Processing Lu")


    path_files_c = glob.glob(os.path.join(transcripts_path,'Mandarin/Lu/*.cha'))
    for file in path_files_c:
        id = str(int(file.split('/')[-1].replace('.cha','')))
        age, gender = get_metadata(file)
        if age != '':
            labels_data.loc[(labels_data['study'] == 'lu') & (labels_data['id'] == id) & (labels_data['language']=='mandarin'), 'age'] = age
        labels_data.loc[(labels_data['study'] == 'lu') & (labels_data['id'] == id) & (labels_data['language']=='mandarin'), 'gender'] = gender

    #ye
    print("Processing Ye")

    path_files_c = glob.glob(os.path.join(transcripts_path,'Mandarin/Ye/*.cha'))
    for file in path_files_c:
        id = file.split('/')[-1].replace('.cha','')
        age, gender = get_metadata(file)
        if age != '':
            labels_data.loc[(labels_data['study'] == 'ye') & (labels_data['id'] == id) & (labels_data['language']=='mandarin'), 'age'] = age
        labels_data.loc[(labels_data['study'] == 'ye') & (labels_data['id'] == id) & (labels_data['language']=='mandarin'), 'gender'] = gender

    #Jalvingh
    print("Processing Jalvingh")

    gender_info = {'FTD01':"male",
               'FTD02':"female",
               'FTD03':"female",
               'FTD04':"female",
               'PPA_LPA':"male",
               'PPA_PnfA':"male",
               'PPA_SD':"female",
               'Park_Dement01':"female",
               'Park_Dement02':"male",
               'Park_Dement03':"male",
               'Park_MCI01':"male",
               'Park_MCI02':"female",
               'Park_MCI03':"male",
               'Park_MCI04':"male"}
    
    for key, value in gender_info.items():
        labels_data.loc[(labels_data['study'] == 'jalvingh') & (labels_data['id'] == key),'gender'] = value

    path_files_c = glob.glob(os.path.join(transcripts_path,'German/Jalvingh/*.cha'))

    for file in path_files_c:
        id = file.split('/')[-1].replace('.cha','')
        age, _ = get_metadata(file)
        print(id,age)
        if age != '':
            labels_data.loc[(labels_data['study'] == 'jalvingh') & (labels_data['id'] == id), 'age'] = age
    
    print("Harmonizing metadata")

    # Harmonize metadata
    labels_data['gender'].replace({'W':"Female", 'M':"Male", 'F':"Female", 'female':"Female", 'male':"Male"}, inplace=True)
    labels_data['age'] = pd.to_numeric(labels_data['age'], errors='coerce').astype('Int64')
    labels_data['education'].replace({'BA':16, '8th grade': 8, -4:'', 'AEI':16, 'TEI':16, 'Primary School':6,
                                        'High School':12, 'Technical School':12, 'Master':18, 'Lyceum': 12},
                                        inplace=True)
    labels_data['education'] = pd.to_numeric(labels_data['education'], errors='coerce').astype('Int64')
    labels_data.to_csv(os.path.join(file_saving_path, 'all_audios_labels_dem.csv'), index=False)

    #Add demographics to final files for FULL-AUDIOS
    print("Adding demographics to training files")
    metadata = pd.read_csv(os.path.join(training_files, 'full_audio', 'Metadata.csv'))
    dem = pd.read_csv(os.path.join(file_saving_path,'all_audios_labels_dem.csv'))

    metadata = metadata.merge(dem[['unique_id','gender','age','education']], left_on='uid', right_on='unique_id',how='left')
    metadata = metadata.fillna(-1)
    metadata = metadata.drop(columns='unique_id')

    metadata.to_csv(os.path.join(training_files, 'full_audio', 'Metadata.csv'),index=False)

    metadata = pd.read_csv(os.path.join(training_files,'Metadata.csv'))

    metadata['unique_id'] = metadata['uid'].apply(lambda x: int(str(x).split('_')[0]))
    metadata = metadata.merge(dem[['unique_id','gender','age','education']], on='unique_id',how='left')
    metadata = metadata.fillna(-1)
    metadata = metadata.drop(columns='unique_id')
    metadata.to_csv(os.path.join(training_files,'Metadata.csv'),index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Download and process TalkBank dementia data.")
    parser.add_argument('--metadata_path', type=str,default='../../data/metadata', help='Path to metadata files')
    parser.add_argument('--file_saving_path', type=str, default='../../data/interim', help='Where to find or save csv files')
    parser.add_argument('--transcripts_path', type=str, default='../../data/raw/talkbank_dementia_transcripts', help="Path to transcripts files")
    parser.add_argument('--training_files', type=str, default='../../data/processed', help="Path to final training files")
    args = parser.parse_args()
    main(args.metadata_path, args.file_saving_path, args.transcripts_path, args.training_files) 
    




