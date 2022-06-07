import argparse
import jieba
import da.delete as delete
import da.insert as insert
import da.substitution as substitution
import da.pronouce as pronouce
import da.local_paraphrase as local_paraphrase
import random
import logging

def get_comma_dict(filename):
    comma_dict = {}
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip('\n')
        if line not in comma_dict:
            comma_dict.setdefault(line, 1)
    return comma_dict

def get_pron_dict(filename):
    pron_dict = {}
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip('\n')
        fields = line.split('\t')
        if len(fields) == 2:
            pron_dict.setdefault(fields[0], fields[1].split(','))
    return pron_dict

def get_vocab_dict(filename):
    vocab_dict = {}
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip('\n')
        if line not in vocab_dict:
            vocab_dict.setdefault(line, 1)
    return vocab_dict

def get_shape_dict(filename):
    char_shape_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        raw_shape_data = file.readlines()
        for d in raw_shape_data:
            char = d.strip().split(',')[0]
            sim_shape_char = list(d.strip().split(',')[1])
            char_shape_dict[char] = sim_shape_char
    return char_shape_dict

def get_delete_chars(filename):
    delete_chars = []
    with open(filename, 'r', encoding='utf-8') as file:
        delete_char_list = file.readlines()
        for d in delete_char_list:
            delete_chars.append(d.strip())
    return delete_chars

class DataAugmentationByRule(object):
    def __init__(self):
        # da
        self.comma_dict = get_comma_dict('./da/dict/comma_words.txt')
        self.pron_dict = get_pron_dict('./da/dict/prononciation.txt')
        self.vocab_dict = get_vocab_dict('./da/dict/vocab.txt')
        self.char_shape_dict = get_shape_dict('./da/dict/Bakeoff2013_CharacterSet_SimilarShape.txt')
        self.delete_chars = get_delete_chars('./da/dict/delete_char_list.txt')
        self.delete_words = get_delete_chars('./da/dict/delete_word_list.txt')
        # self.bert_vocab = bert_vocab
        # 按原数据集分布增强
        # self.weight = weight #[0.078, 0.027, 0.027, 0.102, 0.024, 0.182, 0.168, 0.204, 0.204, 0.006, 0.027, 0.168]


    def augment_by_candidate(self, augment_data, results_detail=None):
        all_augment_data_list = []
        for sent_idx, (primary_result, result_details) in enumerate(zip(augment_data, results_detail)):
            # 今天我在看了昨天买的书。	今天我在看昨天买的书。
            augment_data_list = []
            augment_data_list.append(primary_result)
            for aug_index, candidates in result_details.items():
                for candidate in candidates:
                    list_primary_result = list(primary_result)
                    if aug_index < len(list_primary_result)-1 and candidate != list_primary_result[aug_index]:
                        list_primary_result[aug_index] = candidate
                    augment_data_list.append(''.join(list_primary_result))
            all_augment_data_list.append(augment_data_list)
        return all_augment_data_list
    
    def augment_by_type(self, sentence, augment_type):
        augment_sent = None
        if augment_type == 'delete':
            sub_type = random.choices(['delete_word', 'delete_char'])[0]
            if sub_type == 'delete_word':
                sent_by_word = list(jieba.cut(''.join(sentence)))
                augment_sent = delete.delete_word(sent_by_word, self.comma_dict)
                return augment_sent
            elif sub_type == 'delete_char':
                sent_by_char = list(sentence)
                augment_sent = delete.delete_char(sent_by_char)
                return augment_sent
        elif augment_type == 'insert':
            sub_type = random.choices(['insert_same_word', 'insert_synonyms_word', 'insert_random_word', 'insert_same_char', 'insert_random_char'])[0]
            if sub_type == 'insert_same_word':
                sent_by_word = list(jieba.cut(''.join(sentence)))
                augment_sent = insert.insert_same_word(sent_by_word)
                return augment_sent
            elif sub_type == 'insert_synonyms_word':
                sent_by_word = list(jieba.cut(''.join(sentence)))
                augment_sent = insert.insert_synonyms_word(sent_by_word)
                return augment_sent
            elif sub_type == 'insert_random_word':
                sent_by_word = list(jieba.cut(''.join(sentence)))
                augment_sent = insert.insert_random_word(sent_by_word, self.delete_words)
                return augment_sent
            elif sub_type == 'insert_same_char':
                sent_by_char = list(sentence)
                augment_sent = insert.insert_same_char(sent_by_char)
                return augment_sent
            elif sub_type == 'insert_random_char':
                sent_by_char = list(sentence)
                augment_sent = insert.insert_random_char(sent_by_char, self.delete_chars)
                return augment_sent
        elif augment_type == 'substitution':
            sub_type = random.choices(['w_sub_syn', 'c_sub_pronounce', 'c_sub_shape'])[0]
            if sub_type == 'w_sub_syn':
                sent_by_word = list(jieba.cut(''.join(sentence)))
                augment_sent = substitution.substitute_synonym_word(sent_by_word)
                return augment_sent
            elif sub_type == 'c_sub_pronounce':
                sent_by_char = list(sentence)
                augment_sent = pronouce.generate_pronouce_sent(sent_by_char, self.comma_dict, self.pron_dict, self.vocab_dict)
                return augment_sent
            elif sub_type == 'c_sub_shape':
                sent_by_char = list(sentence)
                augment_sent = substitution.substitute_shape_char(sent_by_char, self.char_shape_dict)
                return augment_sent
        elif augment_type == 'paraphrase':
            sub_type = random.choices(['w_paraph', 'c_paraph'])[0]
            if sub_type == 'w_paraph':
                sent_by_word = list(jieba.cut(''.join(sentence)))
                augment_sent = local_paraphrase.exchange_word(sent_by_word)
                return augment_sent
            elif sub_type == 'c_paraph':
                sent_by_char = list(sentence)
                augment_sent = local_paraphrase.exchange_char(sent_by_char) 
                return augment_sent
        
    

    def augment(self, sentences):
        augment_sents = []

        for sent_idx, sent in enumerate(sentences):
            type = random.choices(self.population, self.weight)[0]
            sent_by_char = list(sent)
            sent_by_word = list(jieba.cut(''.join(sent)))

            augment_sent = None
            if type == 'w_del':
                augment_sent = delete.delete_word(sent_by_word, self.comma_dict)
            if type == 'w_ins_same':
                augment_sent = insert.insert_same_word(sent_by_word)
            if type == 'w_ins_syn':
                augment_sent = insert.insert_synonyms_word(sent_by_word)
            if type == 'w_ran_insert':
                augment_sent = insert.insert_random_word(sent_by_word, self.delete_words)
            if type == 'w_sub_syn':
                augment_sent = substitution.substitute_synonym_word(sent_by_word)
            if type == 'w_paraph':
                augment_sent = local_paraphrase.exchange_word(sent_by_word)
            if type == 'c_del':
                augment_sent = delete.delete_char(sent_by_char)
            if type == 'c_ins_same':
                augment_sent = insert.insert_same_char(sent_by_char)
            if type == 'c_ran_ins':
                augment_sent = insert.insert_random_char(sent_by_char, self.delete_chars)
            if type == 'c_sub_pronounce':
                augment_sent = pronouce.generate_pronouce_sent(sent_by_char, self.comma_dict, self.pron_dict, self.vocab_dict)
            if type == 'c_sub_shape':
                augment_sent = substitution.substitute_shape_char(sent_by_char, self.char_shape_dict)
            if type == 'c_paraph':
                augment_sent = local_paraphrase.exchange_char(sent_by_char)
            if augment_sent:
                augment_sents.append(augment_sent)
        return augment_sents