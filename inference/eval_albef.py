import argparse
import numpy as np
import random
import json
import pandas as pd
from sklearn import metrics
import copy
import sys

sys.path.append('../MM_Claims/')
from utils.loaders import *
from utils.preprocessing import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Evaluted finetuned ALBEF')
parser.add_argument('--dir', type=str, default='data/images_labeled',
                    help='Location of images')
parser.add_argument('--cls', type=int, default=2,
                    help='2 | 3')
parser.add_argument('--model', type=str, default='models/mmc_albef_2cls_wrc.pth',
                    help='Location of images')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model, data_loader):
    # test
    model.eval()

    all_preds = []
    all_labels = []

    for images, text, targets in data_loader:
        
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)   
        text_inputs = tokenizer(text, padding='max_length', max_len=96, truncation=True, return_tensors="pt").to(device)  
        
        prediction = model(images, text_inputs, targets=targets, train=False)  

        preds = torch.argmax(prediction.data, 1)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(targets.cpu().numpy().flatten())
                
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    avg_acc = metrics.accuracy_score(all_labels, all_preds)
    mac_f1 = metrics.f1_score(all_labels, all_preds, average='macro')

    return avg_acc, mac_f1


## Load model
checkpoint = torch.load(args.model)
model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer, ncls=2)
model.load_state_dict(checkpoint["checkpoint"])
model = model.to(device)
model.eval() 

text_dict = json.load(open('data/labeled_text.json', 'r', encoding='utf-8'))
text_processor = get_text_processor(word_stats='twitter', keep_hashtags=True)
text_dict = {twt_id: process_tweet(text_dict[twt_id]["tweet_text"], text_processor) for twt_id in text_dict}

test_wrc_df = pd.read_csv('data/test_with_resolved_conflicts.csv', dtype=str)
test_woc_df = pd.read_csv('data/test_without_conflicts.csv', dtype=str)

te_data_wrc = MMCDataset(args.dir, text_dict, test_wrc_df, args.cls, img_transforms['test'])
te_loader_wrc = DataLoader(te_data_wrc, batch_size=32, num_workers=2)

te_data_woc = MMCDataset(args.dir, text_dict, test_woc_df, args.cls, img_transforms['test'])
te_loader_woc = DataLoader(te_data_woc, batch_size=32, num_workers=2)

## Evaluate on Test Sets
print("Best Epoch %d"%(checkpoint["best_epoch"]))
test_acc, test_f1 = evaluate(model, te_loader_wrc)
print('WRC Test : %.2f, M-F1: %.2f'%(test_acc*100, test_f1*100))

test_acc, test_f1 = evaluate(model, te_loader_woc)
print('WOC Test : %.2f, M-F1: %.2f'%(test_acc*100, test_f1*100))