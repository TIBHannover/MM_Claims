import re

import torch
import math

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

def process_tweet(tweet, text_processor):

    proc_tweet = text_processor.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,!:()|'\"?%\-\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    clean_tweet = " ".join(clean_tweet)
    clean_tweet = re.sub('\s+', ' ', clean_tweet)
    clean_tweet = re.sub(' [.]', '.', clean_tweet)
    clean_tweet = re.sub(' [,]', ',', clean_tweet)
    clean_tweet = re.sub(' [?]', '?', clean_tweet)
    clean_tweet = re.sub('[(] ', '(', clean_tweet)
    clean_tweet = re.sub(' [)]', ')', clean_tweet)
    clean_tweet = re.sub(" [-] ", "-", clean_tweet)

    return clean_tweet


def get_text_processor(word_stats='twitter', keep_hashtags=False):
    return TextPreProcessor(
            # terms that will be normalized , 'number','money', 'time','date', 'percent' removed from below list
            normalize=['url', 'email', 'phone', 'user'],
            # terms that will be annotated
            annotate={"hashtag","allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter=word_stats,

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector=word_stats,

            unpack_hashtags=keep_hashtags,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )



def clip_tokenize(text, _tokenizer, context_length: int = 77):

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + _tokenizer.encode(text)[:75] + [eot_token]
    result = torch.zeros(context_length, dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens)

    return result.unsqueeze(0)


def clip_tokenize_ocr(text, _tokenizer, context_length: int = 77):

    all_tokens = _tokenizer.encode(text)
    k = min(math.ceil(len(all_tokens)/75.0), 2)

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_results = []
    for i in range(k):
        tokens = [sot_token] + all_tokens[i*75: (i+1)*75] + [eot_token]
        result = torch.zeros(context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)

        all_results.append(result.unsqueeze(0))
    
    return all_results