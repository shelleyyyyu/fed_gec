import argparse
import jieba
import delete as delete
import insert as insert
import substitution as substitution
import pronouce as pronouce
import local_paraphrase as local_paraphrase
import random

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment_file', type=str)
    parser.add_argument('--augment_count', type=int, default=10)
    parser.add_argument('--save_file', type=str)
    return parser.parse_args()

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

def main():
    args = parse_config()
    to_augment_dlist = {}
    origin_dlist = []
    with open(args.augment_file, 'r', encoding='utf-8') as file:
        raw_data = file.readlines()

        for d in raw_data:
            if len(d.strip().split('\t')) != 2:
                continue
            incorrect = d.strip().split('\t')[0]
            correct = d.strip().split('\t')[1]
            origin_dlist.append(incorrect+'\t'+correct+'\n')
            if correct not in to_augment_dlist:
                tokenized_sent = ' '.join(jieba.cut(correct))
                to_augment_dlist[correct] = tokenized_sent


    comma_dict = get_comma_dict('./dict/comma_words.txt')
    pron_dict = get_pron_dict('./dict/prononciation.txt')
    vocab_dict = get_vocab_dict('./dict/vocab.txt')

    char_pronounce_dict = {}
    char_shape_dict = {}

    with open('./dict/Bakeoff2013_CharacterSet_SimilarPronunciation.txt', 'r', encoding='utf-8') as file:
        raw_pronounce_data = file.readlines()
        for d in raw_pronounce_data[1:]:
            zh_char = d.split('\t')[0]
            sameyin_samediao = d.split('\t')[1]
            sameyin_yidiao = d.split('\t')[2]
            jingyin_samediao = d.split('\t')[3]
            jingyin_yidiao = d.split('\t')[4]
            # won't consider this
            sameyin_samestrokenum = d.split('\t')[5]
            tmp_dict = {}
            tmp_dict['sameyin_samediao'] = list(sameyin_samediao)
            tmp_dict['sameyin_yidiao'] = list(sameyin_yidiao)
            tmp_dict['jingyin_samediao'] = list(jingyin_samediao)
            tmp_dict['jingyin_yidiao'] = list(jingyin_yidiao)
            char_pronounce_dict[zh_char] = tmp_dict

    with open('./dict/Bakeoff2013_CharacterSet_SimilarShape.txt', 'r', encoding='utf-8') as file:
        raw_shape_data = file.readlines()
        for d in raw_shape_data:
            char = d.strip().split(',')[0]
            sim_shape_char = list(d.strip().split(',')[1])
            char_shape_dict[char] = sim_shape_char

    delete_chars = []
    with open('/Users/shelly/Desktop/laiye/project/gec/code/TtT/data_augmentation_rule/dict/delete_char_list.txt', 'r', encoding='utf-8') as file:
        delete_char_list = file.readlines()
        for d in delete_char_list:
            delete_chars.append(d.strip())

    delete_words = []
    with open('/Users/shelly/Desktop/laiye/project/gec/code/TtT/data_augmentation_rule/dict/delete_word_list.txt', 'r', encoding='utf-8') as file:
        delete_word_list = file.readlines()
        for d in delete_word_list:
            delete_words.append(d.strip())


    population = ['w_del', 'w_ins_same', 'w_ins_syn', 'w_sub_syn', 'w_paraph', 'c_del',
                  'c_ins_same', 'c_sub_pronounce', 'c_sub_shape', 'c_paraph',
                  'w_ran_insert', 'c_ran_insert']
    # NLPCC
    # weight = [0.11, 0.065, 0.065, 0.235, 0.025, 0.11, 0.13, 0.1175, 0.1175, 0.025, 0.0, 0.0]
    # 只增强 Insert
    # weight = [0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # 只增强 Substitution
    # weight = [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0]
    # 只增强 Local Paraphrasing
    # weight = [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
    # 只增强 Delete
    weight = [0.0, 0.13, 0.13, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.13, 0.8]
    # 按原数据集分布增强
    # weight = [0.078, 0.027, 0.027, 0.102, 0.024, 0.182, 0.168, 0.204, 0.204, 0.006, 0.027, 0.168]
    # 平均增强
    # weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


    with open(args.save_file, 'w', encoding='utf=8') as w_file:
        for key, value in to_augment_dlist.items():
            for i in range(args.augment_count):
                type = random.choices(population, weight)[0]
                augment_sent = None
                value = value.split(' ')
                key = list(key)
                if type == 'w_del':
                    augment_sent = delete.delete_word(value, comma_dict=comma_dict)
                if type == 'w_ins_same':
                    augment_sent = insert.insert_same_word(value)
                if type == 'w_ins_syn':
                    augment_sent = insert.insert_synonyms_word(value)
                if type == 'w_ran_insert':
                    augment_sent = insert.insert_random_word(value, delete_words)
                if type == 'w_sub_syn':
                    augment_sent = substitution.substitute_synonym_word(value)
                if type == 'w_paraph':
                    augment_sent = local_paraphrase.exchange_word(value)
                if type == 'c_del':
                    augment_sent = delete.delete_char(key)
                if type == 'c_ins_same':
                    augment_sent = insert.insert_same_char(key)
                if type == 'c_ran_ins':
                    augment_sent = insert.insert_random_char(key, delete_chars)
                if type == 'c_sub_pronounce':
                    augment_sent = pronouce.generate_pronouce_sent(key, comma_dict, pron_dict, vocab_dict)
                if type == 'c_sub_shape':
                    augment_sent = substitution.substitute_shape_char(key, char_shape_dict)
                if type == 'c_paraph':
                    augment_sent = local_paraphrase.exchange_char(key)
                if augment_sent and augment_sent != '' and augment_sent != ' ':
                    w_file.write(augment_sent + '\t' + ''.join(key) + '\n')
                augment_sent = pronouce.generate_pronouce_sent(key, comma_dict, pron_dict, vocab_dict)
                print(key)
                print(augment_sent)
                exit()
        for data in origin_dlist:
            w_file.write(data)

if __name__ == "__main__":
    main()