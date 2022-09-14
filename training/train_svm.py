from sklearn import preprocessing, metrics
from sklearn import decomposition
import pandas as pd
import numpy as np

from thundersvm import *

import json
import pickle

import argparse

parser = argparse.ArgumentParser(description='Training SVM with Image and Text Features')
parser.add_argument('-n','--ncls', type=int, default=2,
                    help='2 | 3')
parser.add_argument('-d','--dtype', type=str, default='wrc',
                    help='wrc | woc')
parser.add_argument('-m','--model', type=str, default='clip',
                    help='clip | albef')
parser.add_argument('-c','--clip', type=str, default='rn50',
                    help='rn50 | rn504 | vit16')
args = parser.parse_args()


def return_metrics(labels, preds):
    acc, f1 =  round(metrics.accuracy_score(labels, preds)*100,2), \
                    round(metrics.f1_score(labels, preds, average='macro')*100,2),

    return acc, f1

def get_best_svm_model(feature_vector_train, label_tr, feature_vector_valid, label_vl):
    param_grid = [{'kernel':'rbf', 'C': np.logspace(-1, 1, 15),
                  'gamma': np.logspace(-1, 1, 15)}]

    pca_list = [1.00,0.99,0.98,0.97,0.96,0.95]
    best_acc = 0.0
    best_model = 0
    best_fsc = 0.0
    best_pca_nk = 0
    temp_xtrain = feature_vector_train
    temp_xval = feature_vector_valid
    for pca_nk in pca_list:
        print(pca_nk)
        if pca_nk != 1.0:
            pca = decomposition.PCA(n_components=pca_nk).fit(temp_xtrain)
            feature_vector_train = pca.transform(temp_xtrain)
            feature_vector_valid = pca.transform(temp_xval)
        for params in param_grid:
            for C in params['C']:
                for gamma in params['gamma']:
                    # Model with different parameters
                    model = SVC(C=C, gamma=gamma, kernel=params['kernel'], random_state=42, 
                                    class_weight='balanced', max_iter=1000)

                    # fit the training dataset on the classifier
                    model.fit(feature_vector_train, label_tr)

                    preds = model.predict(feature_vector_valid)
                    # predict the acc on validation dataset
                    acc, fsc = return_metrics(label_vl, preds)

                    if round(acc,4) >= round(best_acc,4):
                        best_acc = acc
                        best_model = model
                        best_pca_nk = pca_nk
                        best_fsc = fsc

    return best_acc, best_fsc, best_pca_nk, best_model


model = args.model
clip = args.clip
ncls = args.ncls

split_type = 'with_resolved_conflicts' if args.dtype == 'wrc' else 'without_conflicts'
train_df = pd.read_csv('data/train_%s.csv'%(split_type))
val_df = pd.read_csv('data/val_%s.csv'%(split_type))
test_wrc_df = pd.read_csv('data/test_with_resolved_conflicts.csv')
test_woc_df = pd.read_csv('data/test_without_conflicts.csv')

if ncls == 2:
    lab_train, lab_val, lab_test_wrc, lab_test_woc = train_df['claim_binary'].to_numpy(dtype=int), \
            val_df['claim_binary'].to_numpy(dtype=int), test_wrc_df['claim_binary'].to_numpy(dtype=int), \
            test_woc_df['claim_binary'].to_numpy(dtype=int)
else:
    lab_train, lab_val, lab_test_wrc, lab_test_woc = train_df['claim_three'].to_numpy(dtype=int), \
            val_df['claim_three'].to_numpy(dtype=int), test_wrc_df['claim_three'].to_numpy(dtype=int), \
            test_woc_df['claim_three'].to_numpy(dtype=int)


print("----------------- Number of classes:", ncls,"\tModel:", model, "\tCLIP model:", clip if model == 'clip' else 'vit', "\tTrain split type:", split_type, '-----------------\n')

all_feats = json.load(open('features/%s_%s.json'%(model, clip if model == 'clip' else 'vit'), 'r'))
if model == 'clip':
    ft_train = np.column_stack((np.array([all_feats['img_feats'][str(idx)] for idx in train_df['tweet_id']]),
                    np.array([all_feats['text_feats'][str(idx)] for idx in train_df['tweet_id']])))
    ft_val = np.column_stack((np.array([all_feats['img_feats'][str(idx)] for idx in val_df['tweet_id']]),
                    np.array([all_feats['text_feats'][str(idx)] for idx in val_df['tweet_id']])))
    ft_test_wrc = np.column_stack((np.array([all_feats['img_feats'][str(idx)] for idx in test_wrc_df['tweet_id']]),
                    np.array([all_feats['text_feats'][str(idx)] for idx in test_wrc_df['tweet_id']])))
    ft_test_woc = np.column_stack((np.array([all_feats['img_feats'][str(idx)] for idx in test_woc_df['tweet_id']]),
                    np.array([all_feats['text_feats'][str(idx)] for idx in test_woc_df['tweet_id']])))
else:
    ft_train = np.array([all_feats['last'][str(idx)] for idx in train_df['tweet_id']])
    ft_val = np.array([all_feats['last'][str(idx)] for idx in val_df['tweet_id']])
    ft_test_wrc = np.array([all_feats['last'][str(idx)] for idx in test_wrc_df['tweet_id']])
    ft_test_woc = np.array([all_feats['last'][str(idx)] for idx in test_woc_df['tweet_id']])


## Normalize features
ft_train = preprocessing.normalize(ft_train, axis=1)
ft_val = preprocessing.normalize(ft_val, axis=1)
ft_test_wrc = preprocessing.normalize(ft_test_wrc, axis=1)
ft_test_woc = preprocessing.normalize(ft_test_woc, axis=1)

print('Number of train features and labels:', ft_train.shape, lab_train.shape)
print('Number of valiation features and labels:', ft_val.shape, lab_val.shape)
print('Number of test features and labels with resolved label conflicts:', ft_test_wrc.shape, lab_test_wrc.shape)
print('Number of test features and labels wihtout label conflicts:', ft_test_woc.shape, lab_test_woc.shape)


accuracy, f1_score, best_pca_nk, classifier = get_best_svm_model(ft_train, lab_train, ft_val, lab_val)

if best_pca_nk != 1.0:
    pca = decomposition.PCA(n_components=best_pca_nk).fit(ft_train)
    ft_train = pca.transform(ft_train)
    ft_val = pca.transform(ft_val)
    ft_test_wrc = pca.transform(ft_test_wrc)
    ft_test_woc = pca.transform(ft_test_woc)
else:
    pca = {}

test_preds_wrc = classifier.predict(ft_test_wrc)
test_preds_woc = classifier.predict(ft_test_woc)
val_preds = classifier.predict(ft_val)
train_preds = classifier.predict(ft_train)

print('\nPCA No. Components: %.2f, Dim: %d, SV: %d'%(best_pca_nk, ft_val.shape[1], len(classifier.support_)))
print('C: %.3f, Gamma: %.3f, kernel: %s\n'%(classifier.C, classifier.gamma, classifier.kernel))
print('Train Acc/F1: %.2f/%.2f'%(return_metrics(lab_train, train_preds)))
print('Val Acc/F1: %.2f/%.2f'%(return_metrics(lab_val, val_preds)))
print('Test with resolved conflicts Acc/F1: %.2f/%.2f'%(return_metrics(lab_test_wrc, test_preds_wrc)))
print('Test without conflicts Acc/F1: %.2f/%.2f'%(return_metrics(lab_test_woc, test_preds_woc)))
print('\n')



classifier.save_to_file('models/%s_%dncls_%s_%s_svm.m'%(args.dtype, ncls, model, clip if model == 'clip' else 'vit'))
pickle.dump(pca, open('models/%s_%dncls_%s_%s_pca.pkl'%(args.dtype, ncls, model, clip if model == 'clip' else 'vit'), 'wb'))