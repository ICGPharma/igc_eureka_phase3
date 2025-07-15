import pandas as pd
import argparse
import os 

def main(partition_file_path, processed_files_path):

    df = pd.read_csv(partition_file_path)
    training_metadata_file_segments = pd.read_csv(os.path.join(processed_files_path, 'Metadata.csv'))

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

    training_metadata_file_segments = training_metadata_file_segments.merge(df[['segment_id','task']], left_on='uid', right_on='segment_id',how='left')
    training_metadata_file_segments.drop(columns=['segment_id'], inplace=True)

    df.to_csv(partition_file_path, index=False)
    training_metadata_file_segments.to_csv(os.path.join(processed_files_path, 'Metadata.csv'), index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Download and process TalkBank dementia data.")
    parser.add_argument('--partition_file_path', type=str, default='../../data/processed/split_audios_final_partition.csv', help='Path to split data file')
    parser.add_argument('--processed_files_path', type=str,default='../../data/processed', help='Path to save the processed file')
    args = parser.parse_args()

    main(args.partition_file_path, args.processed_files_path)