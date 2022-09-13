import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import os, sys
import numpy as np
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('../MM_Claims/')
from utils.preprocessing import *

import argparse

parser = argparse.ArgumentParser(description='Extract CLIP features')
parser.add_argument('-c','--clip', type=str, default='rn50',
                    help='rn50 | rn504 | vit16')

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_tokenizer = _Tokenizer()

model_nms = {'vit16':'ViT-B/16', 'rn50':'RN50', 'rn504':'RN50x4'}
model_dim = {'vit16': 512, 'rn50': 1024, 'rn504': 640}

## Arguments
clip_nm = args.clip

print('-----------------', 'Extracting features from:', model_nms[clip_nm],'-----------------')

text_processor = get_text_processor(word_stats='twitter', keep_hashtags=True)

im_names, img_feats, text_feats = [], [], []

model, img_preprocess = clip.load(model_nms[clip_nm], device=device)
model.eval()

for phase in ['train', 'val', 'test']:
    text_dict = json.load(open('data/%s_text.json'%(phase), 'r', encoding='utf-8'))
    for key in tqdm(text_dict.keys()):
        text = text_dict[key]['tweet_text']
        img = Image.open(os.path.join('data/images_labeled/', key+'.jpg'))
        
        proc_text = process_tweet(text, text_processor)

        image_tensor = img_preprocess(img).unsqueeze(0).to(device)
        text_tokens = clip_tokenize(proc_text, _tokenizer).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
        
            img_feats.append(image_features.cpu().numpy().flatten().tolist())
            text_feats.append(text_features.cpu().numpy().flatten().tolist())
            im_names.append(key)


feat_dict = {}
feat_dict['img_feats'] = {name: feat for name, feat in zip(im_names, img_feats)}
feat_dict['text_feats'] = {name: feat for name, feat in zip(im_names, text_feats)}
print(len(feat_dict['img_feats']), len(feat_dict['text_feats']))
json.dump(feat_dict, open('features/clip_%s.json'%(clip_nm),'w'))
