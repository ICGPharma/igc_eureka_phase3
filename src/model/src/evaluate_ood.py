import os 
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, precision_recall_curve, auc, average_precision_score
from utils.utils import load_config
import numpy as np
import matplotlib.pyplot as plt

column_map = {'diagnosis_control':0,'diagnosis_mci':1,'diagnosis_adrd':2,'diagnosis_other':3}

def evaluate_loss_ood(config):
    predicted = pd.read_csv(os.path.join(config['paths']['results_path'],"submission_whisper_final.csv"))
    predicted['unique_id'] = predicted['uid'].apply(lambda x: str(x).split('_')[0])
    predicted = predicted[['diagnosis_control','diagnosis_mci','diagnosis_adrd','diagnosis_other','unique_id']].groupby('unique_id').mean()
    predicted = predicted.reset_index()

    test_labels = pd.read_csv(config['data']['labels_file'])
    test_labels['unique_id'] = test_labels['uid'].apply(lambda x: str(x).split('_')[0])
    test_labels = test_labels[['diagnosis_control','diagnosis_mci','diagnosis_adrd','diagnosis_other','unique_id']].groupby('unique_id').first()
    test_labels = test_labels.reset_index()
    test_labels = test_labels[test_labels["unique_id"].isin(predicted["unique_id"])]

    with open(os.path.join(config['paths']['results_path'],"inference_ood.txt"),'a') as file:
        predicted = predicted.drop(columns= ['unique_id'])
        test_labels = test_labels.drop(columns= ['unique_id'])

        loss = log_loss(test_labels.values, predicted.values)
        print(f"Loss-Talkbank: {loss}",file=file)

def get_accuracy_ood(config, labels):

    base_path = config['paths']['results_path']

    results = pd.read_csv(os.path.join(base_path,'submission_whisper_final.csv'))

    results['unique_id'] = results['uid'].apply(lambda x: str(x).split('_')[0])
    results = results.drop(columns='uid')
    results = results.groupby('unique_id').mean()
    results['diagnosis'] = results.idxmax(axis=1).map(column_map)
    filtered_labels = labels[labels.index.isin(results.index)]

    with open(os.path.join(base_path,"accuracy_ood.txt"),'a') as file:
        acc_0 = accuracy_score(filtered_labels[filtered_labels['diagnosis']==0]['diagnosis'],results[filtered_labels['diagnosis']==0]['diagnosis'])
        print(f"Accuracy Control: {acc_0}",file=file)
        acc_1 = accuracy_score(filtered_labels[filtered_labels['diagnosis']==1]['diagnosis'],results[filtered_labels['diagnosis']==1]['diagnosis'])
        print(f"Accuracy MCI: {acc_1}",file=file)
        acc_2 = accuracy_score(filtered_labels[filtered_labels['diagnosis']==2]['diagnosis'],results[filtered_labels['diagnosis']==2]['diagnosis'])
        print(f"Accuracy ADRD: {acc_2}",file=file)
        acc_3 = accuracy_score(filtered_labels[filtered_labels['diagnosis']==3]['diagnosis'],results[filtered_labels['diagnosis']==3]['diagnosis'])
        print(f"Accuracy Other: {acc_3}",file=file)

def _compute_and_plot_auprc(y_true, y_pred_proba, save_path, average='macro'):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n_classes = y_pred_proba.shape[1]

    y_true_bin = np.eye(n_classes)[y_true]

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    auprc_scores = {}
    diagnosis = ['Control','MCI','ADRD','Other']

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        auprc = auc(recall, precision)
        auprc_scores[f'class_{diagnosis[i]}'] = auprc

        plt.plot(recall, precision, color=colors[i % len(colors)],
                 label=f"{diagnosis[i]} (AUPRC = {auprc:.3f})")

    average_score = average_precision_score(y_true_bin, y_pred_proba, average=average)
    auprc_scores['average'] = average_score

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Average AUPRC: {average_score:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    return auprc_scores

def get_auprc_ood(config, labels):
    base_path = config['paths']['results_path']

    results = pd.read_csv(os.path.join(base_path,'submission_whisper_final.csv'))
    results['unique_id'] = results['uid'].apply(lambda x: str(x).split('_')[0])
    results = results.drop(columns='uid')
    results = results.groupby('unique_id').mean()
    results['diagnosis'] = results.idxmax(axis=1).map(column_map)
    filtered_labels = labels[labels.index.isin(results.index)]

    _compute_and_plot_auprc(
        filtered_labels['diagnosis'],
        results[['diagnosis_control','diagnosis_mci','diagnosis_adrd','diagnosis_other']],
        os.path.join(base_path,'auprc_ood.png')
    )

def main():
    config = load_config("../config/config.yaml")

    labels = pd.read_csv(config['data']['labels_file'])
    labels['unique_id'] = labels['uid'].apply(lambda x: str(x).split('_')[0])
    labels = labels.drop(columns='uid')
    labels = labels.groupby('unique_id').first()
    labels['diagnosis'] = labels.idxmax(axis=1).map(column_map)

    evaluate_loss_ood(config)
    get_accuracy_ood(config, labels)
    get_auprc_ood(config, labels)
    
    
if __name__ == '__main__': 
    main()