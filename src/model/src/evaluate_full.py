import os 
import pandas as pd
from sklearn.metrics import log_loss, f1_score
from utils.utils import load_config

def evaluate_talkbank(config,key_id):
    predicted = pd.read_csv(os.path.join(config['paths']['results_path'],"submission_whisper_final.csv"))
    test_labels = pd.read_csv(config['data']['test_labels_talkbank'])
    ood_data = pd.read_csv(config['data']['test_labels_ood'])
    ood_mandarin = ood_data[ood_data['language']=='mandarin'][key_id]
    ood_alexa = ood_data[ood_data['study']=='vas'][key_id]

    with open(os.path.join(config['paths']['results_path'],"inference_chunks.txt"),'a') as file:
        mandarin_pred = predicted[predicted['uid'].isin(ood_mandarin)].drop(columns= ['uid'])
        mandarin_labels = test_labels[test_labels['uid'].isin(ood_mandarin)].drop(columns= ['uid'])
        loss = log_loss(mandarin_labels.values, mandarin_pred.values)
        print(f"Loss-Talkbank Mandarin: {loss}",file=file)

        alexa_pred = predicted[predicted['uid'].isin(ood_alexa)].drop(columns= ['uid'])
        alexa_labels = test_labels[test_labels['uid'].isin(ood_alexa)].drop(columns= ['uid'])
        loss = log_loss(alexa_labels.values, alexa_pred.values)
        print(f"Loss-Talkbank Alexa: {loss}",file=file)

        predicted = predicted.drop(columns= ['uid'])
        test_labels = test_labels.drop(columns= ['uid'])

        loss = log_loss(test_labels.values, predicted.values)
        print(f"Loss-Talkbank: {loss}",file=file)

def evaluate_eureka(config):
    predicted = pd.read_csv(os.path.join(config['paths']['results_path'],"submission_whisper_final.csv"))
    test_labels = pd.read_csv(config['data']['test_labels_eureka'])
    predicted = predicted[predicted['uid'].isin(test_labels['uid'])]

    predicted['diagnosis_adrd'] = predicted['diagnosis_adrd']+predicted['diagnosis_other']

    predicted = predicted.drop(columns= ['uid','diagnosis_other'])
    test_labels = test_labels.drop(columns= ['uid'])

    loss = log_loss(test_labels.values, predicted.values)

    with open(os.path.join(config['paths']['results_path'],"inference_chunks.txt"),'a') as file:
        print(f"Loss-Eureka: {loss}",file=file)

def main():
    config = load_config("../config/config.yaml")
    key_id = "unique_id" if 'data_exp1' in config['data']['metadata_file'] else "segment_id"
    evaluate_eureka(config)
    evaluate_talkbank(config,key_id)
    
if __name__ == '__main__': 
    main()