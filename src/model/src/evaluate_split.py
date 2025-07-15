import os 
import pandas as pd
from sklearn.metrics import log_loss, f1_score
from utils.utils import load_config

def evaluate_talkbank(config):
    predicted = pd.read_csv(os.path.join(config['paths']['results_path'],"submission_whisper_final.csv"))
    predicted['unique_id'] = predicted['uid'].apply(lambda x: str(x).split('_')[0])
    predicted = predicted[['diagnosis_control','diagnosis_mci','diagnosis_adrd','diagnosis_other','unique_id']].groupby('unique_id').mean()
    predicted = predicted.reset_index()
    
    test_labels = pd.read_csv(config['data']['test_labels_talkbank'])
    test_labels['unique_id'] = test_labels['uid'].apply(lambda x: str(x).split('_')[0])
    test_labels = test_labels[['diagnosis_control','diagnosis_mci','diagnosis_adrd','diagnosis_other','unique_id']].groupby('unique_id').first()
    test_labels = test_labels.reset_index()
    
    ood_data = pd.read_csv(config['data']['test_labels_ood'])
    ood_mandarin = ood_data[ood_data['language']=='mandarin']['segment_id'].apply(lambda x: str(x).split('_')[0]).unique()
    ood_alexa = ood_data[ood_data['study']=='vas']['segment_id' ].apply(lambda x: str(x).split('_')[0]).unique()

    with open(os.path.join(config['paths']['results_path'],"inference.txt"),'a') as file:
        mandarin_pred = predicted[predicted['unique_id'].isin(ood_mandarin)].drop(columns= ['unique_id'])
        mandarin_labels = test_labels[test_labels['unique_id'].isin(ood_mandarin)].drop(columns= ['unique_id'])
        loss = log_loss(mandarin_labels.values, mandarin_pred.values)
        print(f"Loss-Talkbank Mandarin: {loss}",file=file)

        alexa_pred = predicted[predicted['unique_id'].isin(ood_alexa)].drop(columns= ['unique_id'])
        alexa_labels = test_labels[test_labels['unique_id'].isin(ood_alexa)].drop(columns= ['unique_id'])
        loss = log_loss(alexa_labels.values, alexa_pred.values)
        print(f"Loss-Talkbank Alexa: {loss}",file=file)

        predicted = predicted.drop(columns= ['unique_id'])
        test_labels = test_labels.drop(columns= ['unique_id'])

        loss = log_loss(test_labels.values, predicted.values)
        print(f"Loss-Talkbank: {loss}",file=file)

def evaluate_eureka(config):
    predicted = pd.read_csv(os.path.join(config['paths']['results_path'],"submission_whisper_final.csv"))
    predicted['unique_id'] = predicted['uid'].apply(lambda x: str(x).split('_')[0])
    predicted = predicted[['diagnosis_control','diagnosis_mci','diagnosis_adrd','diagnosis_other','unique_id']].groupby('unique_id').mean()
    predicted = predicted.reset_index()

    test_labels = pd.read_csv("../data/post_data/Test_labels_eureka.csv")
    test_labels['unique_id'] = test_labels['uid'].apply(lambda x: str(x).split('_')[0])
    test_labels = test_labels[['diagnosis_control','diagnosis_mci','diagnosis_adrd','unique_id']].groupby('unique_id').first()
    test_labels = test_labels.reset_index()

    predicted = predicted[predicted['unique_id'].isin(test_labels['unique_id'])]

    predicted['diagnosis_adrd'] = predicted['diagnosis_adrd']+predicted['diagnosis_other']

    predicted = predicted.drop(columns= ['unique_id','diagnosis_other'])
    test_labels = test_labels.drop(columns= ['unique_id'])

    loss = log_loss(test_labels.values, predicted.values)
    with open(os.path.join(config['paths']['results_path'],"inference.txt"),'a') as file:
        print(f"Loss-Eureka: {loss}",file=file)

def main():
    config = load_config("../config/config.yaml")
    evaluate_eureka(config)
    evaluate_talkbank(config)
    
    
if __name__ == '__main__': 
    main()