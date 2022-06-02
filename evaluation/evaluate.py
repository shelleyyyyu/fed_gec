import os
from modules.annotator import Annotator
from modules.tokenizer import Tokenizer
import argparse
from collections import Counter
from tqdm import tqdm
import torch
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC
from compare_m2_for_evaluation import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")


def annotate(sent_list):
    """
    :param line:
    :return:
    """
    source = sent_list[0]
    source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            target = "".join(target.strip().split())
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str


def get_edits(input_sents, granularity='char', multi_cheapest_strategy='first', batch_size=128, device=0):
    tokenizer = Tokenizer(granularity, device)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default(granularity, multi_cheapest_strategy)
    lines = input_sents

    annotations = []
    count = 0
    sentence_set = set()
    sentence_to_tokenized = {}
    for sent_list in lines:
        for idx, sent in enumerate(sent_list):
            sentence_set.add(sent.strip())
    batch = []
    for sent in tqdm(sentence_set):
        count += 1
        if sent:
            batch.append(sent)
        if count % batch_size == 0:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
            batch = []
    if batch:
        results = tokenizer(batch)
        for s, r in zip(batch, results):
            sentence_to_tokenized[s] = r  # Get tokenization map.

    for line in tqdm(lines):
        ret = annotate(line)
        annotations.append(ret.strip())

    return annotations


# hyp_input_sents = [['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。']]
# ref_input_sents = [['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。']]
#
# # granularity = 'char' # 'word'
# # multi_cheapest_strategy = "first" #, "all"
# # batch_size = 128 dev_batch_size
# # device = 0
# hyp_annotations = get_edits(hyp_input_sents, granularity='word', multi_cheapest_strategy='all', batch_size=128, device=0)
# ref_annotations = get_edits(ref_input_sents, granularity='word', multi_cheapest_strategy='all', batch_size=128, device=0)
# result = calculate_score(hyp_annotations, ref_annotations)
# print(result)