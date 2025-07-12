import pandas as pd

#TODO: ADD ARGS for the files.

def main():

    df = pd.read_csv('audios_final_partition.csv')
    df['task'] = ""
    df['task'][df['study']=='ivanova'] = 'Reading'
    df['task'][df['study']=='wls'] = 'Image description'
    df['task'][df['study']=='baycrest'] = 'Discourse tasks' #Cinderella + AphasiaBank discourse tasks 
    df['task'][(df['study']=='pitt') & (df['file_path'].str.contains('cookie', case=False, na=False))] = 'Image description'
    df['task'][(df['study']=='pitt') & (df['file_path'].str.contains('fluency', case=False, na=False))] = 'Word Fluency'
    df['task'][(df['study']=='pitt') & (df['file_path'].str.contains('recall', case=False, na=False))] = 'Story Recall'
    df['task'][(df['study']=='pitt') & (df['file_path'].str.contains('sentence', case=False, na=False))] = 'Sentence Construction'
    df['task'][(df['study']=='chou') & (df['file_path'].str.contains('market', case=False, na=False))] = 'Image description'
    df['task'][(df['study']=='chou') & (df['file_path'].str.contains('park', case=False, na=False))] = 'Image description'
    df['task'][(df['study']=='chou') & (df['file_path'].str.contains('Daddy', case=False, na=False))] = 'Image description'
    df['task'][(df['study']=='lu') & (df['language']=='english')] = 'Image description'
    df['task'][(df['study']=='lu') & (df['language']=='mandarin')] = 'Word Fluency'
    df['task'][df['study']=='ye'] = 'Word Fluency'
    df['task'][df['study']=='vas'] = 'Alexa'
    df['task'][df['study']=='depaul'] = 'Discourse tasks' # Both data sets include these discourse tasks (from the AphasiaBank discourse protocol)
    df['task'][df['study']=='holland'] = 'Discourse tasks' # AphasiaBank: Methods for studying discourse
    df['task'][df['study']=='hopkins'] = 'Discourse tasks' # All transcripts include a Cinderella story telling. Some also include a picture description (circus, see below), counting forward to 30, counting backward from 30, and The Grandfather Passage.
    df['task'][df['study']=='kempler'] =  'Image description' #'Image description + Conversations' # These files contain conversations between the participant and the investigator. Some also include a Cookie Theft picture description. 
    df['task'][df['study']=='lanzi'] = 'Questions'
    df['task'][df['study']=='delaware'] = 'Discourse tasks' # 'Image description + Discourse tasks' #Picture descriptions -- Cookie Theft, Cat Rescue, "Going and Coming" (Norman Rockwell) / Story narrative -- Cinderella* / Procedural narrative -- Peanut butter and jelly sandwich / Personal narrative -- Hometown
    df['task'][df['study']=='jalvingh'] = 'Spontaneus speech'

    df.to_csv('audios_final_partition_task_2.csv', index=False)


if __name__=="__main__":
    main()