# A Dataset for Multimodal Claim Detection in Social Media

This is the official GitHub page for the paper:

> Gullal Singh Cheema, Sherzod Hakimov, Abdul Sittar, Eric Müller-Budack, Christian Otto, and Ralph Ewerth. 2022. “MM-Claims: A Dataset for Multimodal Claim Detection in Social Media.“ *In Findings of the Association for Computational Linguistics: NAACL 2022, pages 962–979, Seattle, United States. Association for Computational Linguistics.*

## ** Update **

The data will be used in the CLEF [Checkthat Challenge 2023](https://checkthat.gitlab.io/). 

[Register](http://clef2023-labs-registration.dei.unipd.it/registrationForm.php) and Participate.

Refined check-worthiness labels and additional test data to be released in the challenge. [Data](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab/-/tree/main/task1)

## Publication, dataset, annotation

The paper is available here: https://aclanthology.org/2022.findings-naacl.72/

Dataset with tweet IDs and labels are available at: https://data.uni-hannover.de/dataset/mm_claims

Annotation guideline document is available here: https://github.com/TIBHannover/MM_Claims/blob/main/misc_files/annotation_doc.pdf

For access to images and tweets, send an email with organization (university/institute) and purpose/usage details to gullal.cheema@tib.eu



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

- Add two changes to `ALBEF/models/model_ve.py` to avoid path errors:
   - At the top:
      ```
         import sys
         sys.path.append('ALBEF/')
      ```
   - `'ALBEF/'+config['bert_config']` in line `bert_config = BertConfig.from_json_file(config['bert_config'])`


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


## Fine-tune ALBEF

`python training/finetune_albef_mm.py --fr_no 8 --bs 8 --cls 2`

## Inference
- Download trained svm models (above) from [here](https://tib.eu/cloud/s/5SK6BzdcfFQbN8A) and move them in `models/`

- Evaluate svm trained with clip features on test splits, Binary claim detection:

   `python inference/eval_svm.py -m clip -c rn504 -d wrc`
   
   Output:
   ```
   ----------------- Number of classes: 2  Model: clip     CLIP model: rn504       Train split type: with_resolved_conflicts -----------------

   Number of test features and labels with resolved label conflicts: (585, 1280) (585,)
   Number of test features and labels wihtout label conflicts: (525, 1280) (525,)

   Test with resolved conflicts Acc/F1: 77.78/77.39
   Test without conflicts Acc/F1: 79.43/78.39
   ```
   
- Evaluate svm trained with albef features on test splits, Binary claim detection:

   `python inference/eval_svm.py -m albef -d wrc`
   
   Output:
   ```
   ----------------- Number of classes: 2  Model: albef    CLIP model: vit         Train split type: with_resolved_conflicts -----------------

   Number of test features and labels with resolved label conflicts: (585, 768) (585,)
   Number of test features and labels wihtout label conflicts: (525, 768) (525,)

   Test with resolved conflicts Acc/F1: 76.92/76.46
   Test without conflicts Acc/F1: 78.67/77.51
   ```
   
 - Evaluate svm trained with albef features on test splits, Tertiary claim detection:
 
   `python inference/eval_svm.py -m albef -n 3 -d woc`
   
   Output:
   ```
   ---------------- Number of classes: 3  Model: albef    CLIP model: vit         Train split type: without_conflicts -----------------

   Number of test features and labels with resolved label conflicts: (585, 768) (585,)
   Number of test features and labels wihtout label conflicts: (525, 768) (525,)

   Test with resolved conflicts Acc/F1: 71.45/58.61
   Test without conflicts Acc/F1: 75.43/55.54
   ```

 - Evaluate albef:

   `python inference/eval_albef.py --cls 2 --model models/mmc_albef_2cls_wrc.pth`
   

If you find the data or the code useful, cite us:
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
