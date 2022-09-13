# A Dataset for Multimodal Claim Detection in Social Media

This is the official GitHub page for the paper:

> Gullal Singh Cheema, Sherzod Hakimov, Abdul Sittar, Eric Müller-Budack, Christian Otto, and Ralph Ewerth. 2022. “MM-Claims: A Dataset for Multimodal Claim Detection in Social Media.“ *In Findings of the Association for Computational Linguistics: NAACL 2022, pages 962–979, Seattle, United States. Association for Computational Linguistics.*

The paper is available here: https://aclanthology.org/2022.findings-naacl.72/

Tweet IDs and labels are available at: https://data.uni-hannover.de/dataset/mm_claims

For access to images and tweets, send an email with organization (university/institute) and purpose details to gullal.cheema@tib.eu


## Environment Setup

- Create conda environment: `conda env create -f environment.yml`
- Activate the environment: `conda activate mmclaim11`
- Install thundersvm:
```
git clone https://github.com/Xtra-Computing/thundersvm.git

cd thundersvm
mkdir build
cd build
cmake ..
make -j

cd python
python setup.py install
```

- Install clip: `pip install git+https://github.com/openai/CLIP.git`


## Data Setup
- Download the training, validation and test split csvs in `data/`
- Download and extract image zip files in `data/`
- Download text jsons in `data/`
- Download pre-trained ALBEF checkpoint from https://github.com/salesforce/ALBEF and move it to `albef_checkpoint/`


## Extract Features
- Extract CLIP features `python extraction/feat_extract_clip.py -c rn504`
- Extract ALBEF features `python extraction/feat_extract_albef.py`

## Training SVM models (Best clip variant from Table 4 in paper)
- Train with clip features on split with resolved label conflicts, Binary claim detection:

   `python training/train_svm.py -n 2 -m clip -c rn504 -d wrc`
   
- Train with clip features on split with resolved label conflicts, Tertiary claim detection:

  `python training/train_svm.py -n 3 -m clip -c rn50 -d wrc`
  
- Train with clip features on split without label conflicts, Tertiary claim detection:
  
  `python training/train_svm.py -n 3 -m clip -c vit16 -d woc`
  
- Replace `-m clip` with `-m albef` to use albef features.


### To-Do
- [ ] Fine-tuning ALBEF script
- [ ] Direct inference with trained models script


If you find the data or the code useful:
```
@inproceedings{DBLP:conf/naacl/CheemaHSMOE22,
  author    = {Gullal Singh Cheema and
               Sherzod Hakimov and
               Abdul Sittar and
               Eric M{\"{u}}ller{-}Budack and
               Christian Otto and
               Ralph Ewerth},
  editor    = {Marine Carpuat and
               Marie{-}Catherine de Marneffe and
               Iv{\'{a}}n Vladimir Meza Ru{\'{\i}}z},
  title     = {MM-Claims: {A} Dataset for Multimodal Claim Detection in Social Media},
  booktitle = {Findings of the Association for Computational Linguistics: {NAACL}
               2022, Seattle, WA, United States, July 10-15, 2022},
  pages     = {962--979},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.findings-naacl.72},
  timestamp = {Mon, 18 Jul 2022 17:13:00 +0200},
  biburl    = {https://dblp.org/rec/conf/naacl/CheemaHSMOE22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
