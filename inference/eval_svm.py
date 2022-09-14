from sklearn import preprocessing, metrics
from sklearn import decomposition
import pandas as pd
import numpy as np

from thundersvm import *

import json
import pickle

import argparse

parser = argparse.ArgumentParser(description='Test SVM with Image and Text Features')
parser.add_argument('-n','--ncls', type=int, default=2,
                    help='2 | 3')
parser.add_argument('-d','--dtype', type=str, default='wrc',
                    help='wrc | woc')
parser.add_argument('-m','--model', type=str, default='clip',
                    help='clip | albef')
parser.add_argument('-c','--clip', type=str, default='rn504',
                    help='rn50 | rn504 | vit16')
args = parser.parse_args()


def return_metrics(labels, preds):
    acc, f1 =  round(metrics.accuracy_score(labels, preds)*100,2), \
                    round(metrics.f1_score(labels, preds, average='macro')*100,2),

    return acc, f1


model = args.model
clip = args.clip
ncls = args.ncls

split_type = 'with_resolved_conflicts' if args.dtype == 'wrc' else 'without_conflicts'
test_wrc_df = pd.read_csv('data/test_with_resolved_conflicts.csv')
test_woc_df = pd.read_csv('data/test_without_conflicts.csv')

if ncls == 2:
    lab_test_wrc, lab_test_woc = test_wrc_df['claim_binary'].to_numpy(dtype=int), \
            test_woc_df['claim_binary'].to_numpy(dtype=int)
else:
    lab_test_wrc, lab_test_woc = test_wrc_df['claim_three'].to_numpy(dtype=int), \
            test_woc_df['claim_three'].to_numpy(dtype=int)


print("----------------- Number of classes:", ncls,"\tModel:", model, "\tCLIP model:", clip if model == 'clip' else 'vit', "\tTrain split type:", split_type, '-----------------\n')

all_feats = json.load(open('features/%s_%s.json'%(model, clip if model == 'clip' else 'vit'), 'r'))
if model == 'clip':
    ft_test_wrc = np.column_stack((np.array([all_feats['img_feats'][str(idx)] for idx in test_wrc_df['tweet_id']]),
                    np.array([all_feats['text_feats'][str(idx)] for idx in test_wrc_df['tweet_id']])))
    ft_test_woc = np.column_stack((np.array([all_feats['img_feats'][str(idx)] for idx in test_woc_df['tweet_id']]),
                    np.array([all_feats['text_feats'][str(idx)] for idx in test_woc_df['tweet_id']])))
else:
    ft_test_wrc = np.array([all_feats['last'][str(idx)] for idx in test_wrc_df['tweet_id']])
    ft_test_woc = np.array([all_feats['last'][str(idx)] for idx in test_woc_df['tweet_id']])


## Normalize features
ft_test_wrc = preprocessing.normalize(ft_test_wrc, axis=1)
ft_test_woc = preprocessing.normalize(ft_test_woc, axis=1)

print('Number of test features and labels with resolved label conflicts:', ft_test_wrc.shape, lab_test_wrc.shape)
print('Number of test features and labels wihtout label conflicts:', ft_test_woc.shape, lab_test_woc.shape)


## Load saved models
classifier = SVC()
classifier.load_from_file('models/%s_%dncls_%s_%s_svm.m'%(args.dtype, ncls, model, clip if model == 'clip' else 'vit'))

pca = pickle.load(open('models/%s_%dncls_%s_%s_pca.pkl'%(args.dtype, ncls, model, clip if model == 'clip' else 'vit'), 'rb'))

if pca:
    ft_test_wrc = pca.transform(ft_test_wrc)
    ft_test_woc = pca.transform(ft_test_woc)


test_preds_wrc = classifier.predict(ft_test_wrc)
test_preds_woc = classifier.predict(ft_test_woc)

print('\nTest with resolved conflicts Acc/F1: %.2f/%.2f'%(return_metrics(lab_test_wrc, test_preds_wrc)))
print('Test without conflicts Acc/F1: %.2f/%.2f'%(return_metrics(lab_test_woc, test_preds_woc)))
print('\n')
