import ruamel.yaml as yaml
import json, os, sys
from PIL import Image
import pandas as pd
from tqdm import tqdm

from torchvision import transforms

sys.path.append('../MM_Claims/')
from utils.preprocessing import *

from ALBEF.models.model_ve import ALBEF
from ALBEF.models.vit import interpolate_pos_embed
from ALBEF.models.tokenization_bert import BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_processor = get_text_processor(word_stats='twitter', keep_hashtags=True)


## Load visual entailment model config file
config = yaml.load(open('ALBEF/configs/VE.yaml', 'r'), Loader=yaml.Loader)

## VE ALBEF specific image transform
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ]) 


##Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#### Model #### 
print('Creating model')
model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer)

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
        
msg = model.load_state_dict(state_dict,strict=False)
print('loaded checkpoint from albef_checkpoint/ALBEF.pth')

model = model.to(device) 
model.eval()
## ALBEF Loaded. Now extract features for the data

all_feat_dict = {'last': {}, 'average': {}}

## Extract features
for phase in ['train', 'val', 'test']:
    text_dict = json.load(open('data/%s_text.json'%(phase), 'r', encoding='utf-8'))

    for key in tqdm(text_dict.keys()):
        text = text_dict[key]['tweet_text']
        image = Image.open(os.path.join('data/images_labeled/', key+'.jpg')).convert('RGB')
        image_input = transform(image).to(device)

        proc_text = process_tweet(text, text_processor)
        text_input = tokenizer(proc_text, return_tensors='pt').to(device)

        image_embeds = model.visual_encoder(image_input.unsqueeze(0)) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image_input.device)

        with torch.no_grad():
            output = model.text_encoder(text_input.input_ids, 
                                            attention_mask = text_input.attention_mask, 
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,  
                                            output_hidden_states=True,      
                                            return_dict = True
                                            )
        
        all_feat_dict['last'][key] = output.last_hidden_state[:,0,:].cpu().numpy().flatten().tolist()
        all_feat_dict['average'][key] = output.last_hidden_state.mean(dim=1).cpu().numpy().flatten().tolist()
        

json.dump(all_feat_dict, open('features/albef_vit.json', 'w'))
