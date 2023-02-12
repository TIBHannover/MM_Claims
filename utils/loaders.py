import os
import sys
import re
from PIL import Image
import ruamel.yaml as yaml

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

sys.path.append('../MM_Claims/')
from utils.mmcALBEF import *
from ALBEF.models.vit import interpolate_pos_embed
from ALBEF.models.tokenization_bert import BertTokenizer

## Load visual entailment model config file
config = yaml.load(open('ALBEF/configs/VE.yaml', 'r'), Loader=yaml.Loader)

## Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

## VE ALBEF specific image transform
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

img_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
        ]),
        'test': transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])}


class MMCDataset(Dataset):
    def __init__(self, img_dir, data_dict, indx_df, ncls, img_transform=None):
        self.img_dir = img_dir
        self.indx_df = indx_df
        self.text_dict = data_dict
        self.img_transform = img_transform
        self.cls = 'claim_binary' if ncls == 2 else 'claim_three'

    def __len__(self):
        return len(self.indx_df)

    def __getitem__(self, idx):
        txt_id = self.indx_df['tweet_id'][idx]
        label = int(self.indx_df[self.cls][idx])
        image = Image.open(os.path.join(self.img_dir, txt_id+'.jpg')).convert('RGB')

        text = self.text_dict[txt_id]
        
        image = self.img_transform(image)

        return image, text, label


def load_albef_model(freeze_layers=8, ncls=2):
    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer, ncls=ncls)

    ### Load pretrained checkpoint
    checkpoint = torch.load('albef_checkpoint/ALBEF.pth', map_location='cpu') 
    state_dict = checkpoint['model']

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    if config['distill']:
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):                
        if 'bert' in key:
            new_key = key.replace('bert.','')
            state_dict[new_key] = state_dict[key] 
            del state_dict[key]
            
    msg = model.load_state_dict(state_dict, strict=False)
    print('loaded checkpoint from albef_checkpoints/ALBEF.pth')

    ## Freeze layers of model
    vec = list(range(0, freeze_layers))
    for name, param in model.visual_encoder.blocks.named_parameters():
        if int(re.findall(r'[0-9]+', name)[0]) in vec:
            param.requires_grad = False
    for name, param in model.visual_encoder.named_parameters():     
        if 'cls_token' in name or 'patch_embed' in name or 'pos_embed' in name:
            param.requires_grad = False
    for param in model.text_encoder.embeddings.parameters():
            param.requires_grad = False
    for name, param in model.text_encoder.encoder.named_parameters():
        if int(re.findall(r'[0-9]+', name)[0]) in vec:
            param.requires_grad = False
    
    return model