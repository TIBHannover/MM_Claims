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
from ALBEF import utils
from ALBEF.scheduler import create_scheduler
from ALBEF.optim import create_optimizer

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Finetuning ALBEF')
parser.add_argument('--dir', type=str, default='data/images_labeled',
                    help='Location of images')
parser.add_argument('--cls', type=int, default=2,
                    help='2 | 3')
parser.add_argument('--dtype', type=str, default='wrc',
                    help='wrc | woc')
parser.add_argument('--bs', type=int, default=8,
                    help='8 | 16')
parser.add_argument('--fr_no', type=int, default=8,
                    help='0-12 | Number of encoder layers to freeze')
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def train(model, epoch):
    model.train()

    total_tr_loss = 0
    step_size = 100
    warmup_iterations = warmup_steps*step_size 

    for idx, batch in enumerate(tr_loader):
        images, text, targets = batch

        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        
        text_inputs = tokenizer(text, padding='max_length', max_len=96, truncation=True, return_tensors="pt").to(device)

        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,idx/len(tr_loader))

        loss, logits = model(images, text_inputs, targets=targets, train=True, alpha=alpha)    
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_tr_loss += loss.item()

        if idx%40 == 0:
            preds = torch.argmax(logits.detach(), 1).cpu().numpy().flatten()
            print('Batch %d of %d, Train Acc: %.4f'%(idx, len(tr_loader), 
                        metrics.accuracy_score(preds, targets.cpu().numpy().flatten())))

        if epoch==0 and idx%step_size==0 and idx<=warmup_iterations: 
            lr_scheduler.step(idx//step_size)

    lr_scheduler.step(epoch+warmup_steps+1)  

    avg_tr_loss = total_tr_loss/len(tr_loader)

    print('Avg. Train loss: %.4f'%(avg_tr_loss))


## Load data and create data loaders
text_dict = json.load(open('data/labeled_text.json', 'r', encoding='utf-8'))
text_processor = get_text_processor(word_stats='twitter', keep_hashtags=True)
text_dict = {twt_id: process_tweet(text_dict[twt_id]["tweet_text"], text_processor) for twt_id in text_dict}

split_type = 'with_resolved_conflicts' if args.dtype == 'wrc' else 'without_conflicts'
train_df = pd.read_csv('data/train_%s.csv'%(split_type), dtype=str)
val_df = pd.read_csv('data/val_%s.csv'%(split_type), dtype=str)
test_wrc_df = pd.read_csv('data/test_with_resolved_conflicts.csv', dtype=str)
test_woc_df = pd.read_csv('data/test_without_conflicts.csv', dtype=str)

print("\n# Training samples: %d"%(len(train_df)))
print("# Validation samples: %d"%(len(val_df)))
print("# WRC Test samples: %d"%(len(test_wrc_df)))
print("# WOC Test samples: %d\n"%(len(test_woc_df)))

tr_data = MMCDataset(args.dir, text_dict, train_df, args.cls, img_transforms['train'])
tr_loader = DataLoader(tr_data, shuffle=True, batch_size=args.bs, num_workers=2)

vl_data = MMCDataset(args.dir, text_dict, val_df, args.cls, img_transforms['test'])
vl_loader = DataLoader(vl_data, batch_size=int(args.bs/2), num_workers=2)

te_data_wrc = MMCDataset(args.dir, text_dict, test_wrc_df, args.cls, img_transforms['test'])
te_loader_wrc = DataLoader(te_data_wrc, batch_size=int(args.bs/2), num_workers=2)

te_data_woc = MMCDataset(args.dir, text_dict, test_woc_df, args.cls, img_transforms['test'])
te_loader_woc = DataLoader(te_data_woc, batch_size=int(args.bs/2), num_workers=2)


## Load model
model = load_albef_model(freeze_layers=args.fr_no, ncls=args.cls)
model = model.to(device)

## Training only these layers
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)


## Optimizers and Schedulers
arg_opt = utils.AttrDict(config['optimizer'])
optimizer = create_optimizer(arg_opt, model)
arg_sche = utils.AttrDict(config['schedular'])
lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

max_epoch = config['schedular']['epochs']
warmup_steps = config['schedular']['warmup_epochs']


## Train
best_epoch = 0
best_vl_acc = 0
best_model = 0

for epoch in range(0, max_epoch):

    print("\n----- Epoch %d/%d ------"%(epoch+1, max_epoch))
      
    train(model, epoch)

    val_acc, mac_f1 =  evaluate(model, vl_loader)

    print("Val Acc: %.2f, M-F1: %.2f\n"%(val_acc*100, mac_f1*100))

    if val_acc >= best_vl_acc:
        best_vl_acc = val_acc
        best_epoch = epoch+1
        best_model = copy.deepcopy(model)


torch.save({
    "checkpoint": best_model.state_dict(),
    "best_epoch": best_epoch,
    "batch_size": args.bs,
    "freeze_layers": args.fr_no
}, "models/mmc_albef_%dcls_%s.pth"%(args.cls, args.dtype))

## Evaluate on Test Sets
print("Best Epoch %d"%(best_epoch))
test_acc, test_f1 = evaluate(best_model, te_loader_wrc)
print('WRC Test : %.2f, M-F1: %.2f'%(test_acc*100, test_f1*100))

test_acc, test_f1 = evaluate(best_model, te_loader_woc)
print('WOC Test : %.2f, M-F1: %.2f'%(test_acc*100, test_f1*100))