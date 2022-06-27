# Author: yxsu
import sys
import torch
import argparse
sys.path.append('..')
from bert import BERTLM
from data import Vocab, SEP, MASK, CLS
from main import myModel
import numpy as np
from funcs import process_batch_tag

import time
mstime = lambda: int(round(time.time() * 1000))    

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters

def init_empty_bert_model(bert_args, bert_vocab, gpu_id):
    bert_ckpt = torch.load('./model/bert/bert.ckpt')
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    return bert_model
    # bert_ckpt= torch.load(args.bert_path)
    # bert_args = bert_ckpt['args']
    # bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    # bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
    #     bert_args.dropout, bert_args.layers, bert_args.approx)
    # bert_model.load_state_dict(bert_ckpt['model'])
    # if torch.cuda.is_available():
    #     bert_model = bert_model.cuda(gpu_id)
    # return bert_model, bert_vocab, bert_args


def init_sequence_tagging_model(empty_bert_model, args, bert_args, gpu_id, bert_vocab, model_parameters):
    number_class = args.number_class
    embedding_size = bert_args.embed_dim
    batch_size = args.batch_size
    dropout = args.dropout
    device = gpu_id
    vocab = bert_vocab
    loss_type = args.loss_type
    model = myModel(empty_bert_model, number_class, embedding_size, batch_size, dropout, device, vocab, loss_type)
    model.load_state_dict(model_parameters)
    return model

def get_tag_mask_matrix(batch_text_list):
    tag_matrix = []
    mask_matrix = []
    batch_size = len(batch_text_list)
    max_len = 0
    for instance in batch_text_list:
        max_len = max(len(instance), max_len)
    max_len += 1 # 1 for [CLS]
    for i in range(batch_size):
        one_tag = list(np.zeros(max_len).astype(int))
        tag_matrix.append(one_tag)
        one_mask = [1]
        one_valid_len = len(batch_text_list[i])
        for j in range(one_valid_len):
            one_mask.append(1)
        len_diff = max_len - len(one_mask)
        for _ in range(len_diff):
            one_mask.append(0)
        mask_matrix.append(one_mask)
        assert len(one_mask) == len(one_tag)
    return np.array(tag_matrix), np.array(mask_matrix)

def join_str(in_list):
    out_str = ''
    for token in in_list:
        out_str += str(token) + ''
    return out_str.strip()

def predict_one_text_split(text_split_list, seq_tagging_model, id_label_dict, label_id_dict, bert_vocab):
    # text_split_list is a list of tokens ['word1', 'word2', ...]
    text_list = [text_split_list]
    # text_list = [['我', '的', '家', '是', '在', '您', '的', '工', '厂', '附', '近', '，', '最', '近', '您', '的', '工', '厂', '得', '噪', '音', '和', '臭', '味', '都', '让', '这', '里', '的', '人', '民', '真', '受', '不', '了', '。', '<-MASK->', '<-SEP->']]
    tag_list = [[bert_vocab.token2idx(w) for w in text] for text in text_list]
    #label_list = [[label_id_dict[text] for text in text_split_list]]
    tag_matrix = process_batch_tag(tag_list, bert_vocab)
    _, mask_matrix = get_tag_mask_matrix(text_list)
    # print(text_list)
    # print(tag_list)
    # print(tag_matrix)
    # print(mask_matrix)
    # print(len(tag_list[0]))
    # print(len(text_list[0]))
    # print(len(tag_matrix[0]))
    # print(len(mask_matrix[0]))
    # exit()
    decode_result, _, _, _, input_data = seq_tagging_model(text_list, mask_matrix, tag_matrix, fine_tune=False)
    dev_text = ''
    for token in text_list[0]:
        dev_text += token + ' '
    valid_dev_text_len = len(text_list[0])
    pred_tags = []
    for tag in decode_result[0][1:valid_dev_text_len + 1]:
        pred_tags.append(int(tag))
    valid_text_len = len(text_split_list)
    valid_decode_result = decode_result[0][1: valid_text_len + 1]

    tag_result = []
    for token in valid_decode_result:
        tag_result.append(id_label_dict[int(token)])
    return tag_result, input_data[1:].t()[0].tolist(), pred_tags
    #return valid_decode_result

def get_text_split_list(text, max_len):
    result_list = []
    text_list = [w for w in text] + [SEP]
    valid_len = len(text_list)
    split_num = (len(text_list) // max_len) + 1
    if split_num == 1:
        result_list = [text_list]
    else:
        b_idx = 0
        e_idx = 1
        for i in range(max_len):
            b_idx = i * max_len
            e_idx = (i + 1) * max_len
            result_list.append(text_list[b_idx:e_idx])
        if e_idx < valid_len:
            result_list.append(text_list[e_idx:])
        else:
            pass
    return result_list

def predict_one_text(text, gold, max_len, seq_tagging_model, id_label_dict, label_id_dict, bert_vocab):
    text_split_list = get_text_split_list(text, max_len)
    gold_split_list = get_text_split_list(gold, max_len)

    #all_text_result = []
    all_decode_result = []
    for idx in range(len(text_split_list)):
        one_decode_result, wrong, predict = predict_one_text_split(text_split_list[idx], seq_tagging_model, id_label_dict, label_id_dict, bert_vocab)
        # print(bert_vocab)
        gold = [bert_vocab.token2idx(token) for token in gold_split_list[idx]]
        #all_text_result.extend(text_split_list[idx])
        all_decode_result.extend(one_decode_result)
    # result_text = join_str(all_text_result)
    tag_predict_result = join_str(all_decode_result)
    return tag_predict_result, wrong, gold, predict

def get_label_dict(label_path):
    label_dict = {}
    with open(label_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            label_id = int(content_list[1])
            label = content_list[0]
            label_dict[label_id] = label
    return label_dict

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--out_path',type=str)
    parser.add_argument('--gpu_id',type=int, default=0)
    parser.add_argument('--max_len',type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    ckpt_path = args.ckpt_path
    test_data = args.test_data
    out_path = args.out_path
    gpu_id = args.gpu_id
    max_len = args.max_len

    print("loading..")
    bert_args, model_args, bert_vocab, model_parameters = extract_parameters(ckpt_path)
    
    id_label_dict = {}
    label_id_dict = {}
    for lid, label in enumerate(bert_vocab._idx2token):
        id_label_dict[lid] = label
        label_id_dict[label] = lid
    print(label_id_dict[CLS])
    
    model_args.number_class = len(id_label_dict)

    empty_bert_model = init_empty_bert_model(bert_args, bert_vocab, gpu_id)
    seq_tagging_model = init_sequence_tagging_model(empty_bert_model, model_args, bert_args, gpu_id, bert_vocab, model_parameters)
    #optimizer = torch.optim.Adam(seq_tagging_model.parameters(), args.lr)

    if torch.cuda.is_available():
        seq_tagging_model.cuda(gpu_id)

    print("eval...")
    seq_tagging_model.eval()

    wrong_tag_list = []
    gold_tag_list = []
    pred_tag_list = []

    with torch.no_grad():
        with open(out_path, 'w', encoding = 'utf8') as o:
            with open(test_data, 'r', encoding = 'utf8') as i:
                start = mstime()
                lines = i.readlines()
                for l in lines:
                    content_list = l.strip().split('\t')
                    text = content_list[0]
                    gold = content_list[1]
                    res, wrong_id_list, gold_id_list, predict_id_list = predict_one_text(text, gold, max_len, seq_tagging_model, id_label_dict, label_id_dict, bert_vocab)
                    res = res.replace(SEP, '').strip()
                    o.writelines(text + "\t" + res + "\t" + gold + "\n")
                    wrong_tag_list.append(wrong_id_list)
                    gold_tag_list.append(gold_id_list)
                    pred_tag_list.append(predict_id_list)

                # Evaluation
                right_true, right_false, wrong_true, wrong_false = 0, 0, 0, 0
                all_right, all_wrong = 0, 0
                print(len(gold_tag_list), len(pred_tag_list), len(wrong_tag_list))

                for glist, plist, wlist in zip(gold_tag_list, pred_tag_list, wrong_tag_list):
                    for c, w, p in zip(glist, wlist, plist):
                        # Right
                        if w == c:
                            if p == c:
                                # TN
                                right_true += 1
                            else:
                                # FP
                                right_false += 1
                        else:  # Wrong
                            if p == c:
                                # TP
                                wrong_true += 1
                            else:
                                # FN
                                wrong_false += 1

                print(wrong_true, wrong_false, right_false, right_true)
                print(wrong_true + wrong_false + right_false + right_true)
                all_wrong = wrong_true + wrong_false
                print(all_wrong)
                r = wrong_true / all_wrong
                p = wrong_true / (right_false + wrong_true)
                f1 = (2 * r * p) / (r + p)
                acc = (right_true + wrong_true) / (right_true + wrong_true + right_false + wrong_false)
                print('Test acc : %.4f, f1 : %.4f, precision : %.4f, recall : %.4f' % (acc, f1, p, r))

