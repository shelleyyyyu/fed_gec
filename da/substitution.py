import random
import synonyms

def substitute_synonym_word(sent, pos_dict=None):
    # 万一您不想改善目前的情况的话 ， 我们会向法院告贵工厂，也要跟媒体说我们住民的困扰。
    recusive_cnt = 0
    tmp_list = list(sent)
    while True and recusive_cnt <= 5:
        recusive_cnt += 1
        word_index_to_substitute = random.sample([i for i in range(len(tmp_list))], 1)[0]
        word_to_substitute = tmp_list[word_index_to_substitute]
        synonyms_list = synonyms.nearby(word_to_substitute, 15)[0]
        synonyms_score = synonyms.nearby(word_to_substitute, 15)[1]
        synonyms_pass_index_list = [idx for idx, (word, score) in enumerate(zip(synonyms_list, synonyms_score)) if score < 0.7 and len(word) == len(word_to_substitute)]
        if len(synonyms_pass_index_list) > 1:
            synonyms_to_substitute = synonyms_list[random.sample(synonyms_pass_index_list, 1)[0]]
            sent = tmp_list[:word_index_to_substitute] + [synonyms_to_substitute] + tmp_list[(word_index_to_substitute+1):]
            return ''.join(sent)
    return None

def substitute_homonym_char(sent, char_pronounce_dict, pos_dict=None):
    recusive_cnt = 0
    tmp_list = list(sent)
    while True and recusive_cnt <= 5:
        recusive_cnt += 1
        char_index_to_substitute = random.sample([i for i in range(len(tmp_list))], 1)[0]
        char_to_substitute = tmp_list[char_index_to_substitute]
        if char_to_substitute in char_pronounce_dict:
            homonyms = char_pronounce_dict[char_to_substitute]['sameyin_samediao'] + \
                       char_pronounce_dict[char_to_substitute]['sameyin_yidiao'] + \
                       char_pronounce_dict[char_to_substitute]['jingyin_samediao'] + \
                       char_pronounce_dict[char_to_substitute]['jingyin_yidiao']

            if len(homonyms) > 1:
                synonyms_to_substitute = random.sample(homonyms, 1)[0]
                sent = tmp_list[:char_index_to_substitute] + [synonyms_to_substitute] + tmp_list[(char_index_to_substitute + 1):]
                return ''.join(sent)
    return None

def substitute_shape_char(sent, char_shape_dict, pos_dict=None):
    recusive_cnt = 0
    tmp_list = list(sent)
    while True and recusive_cnt <= 5:
        recusive_cnt += 1
        char_index_to_substitute = random.sample([i for i in range(len(tmp_list))], 1)[0]
        char_to_substitute = tmp_list[char_index_to_substitute]
        if char_to_substitute in char_shape_dict:
            shape_list = char_shape_dict[char_to_substitute]
            if len(shape_list) > 1:
                shape_to_substitute = random.sample(shape_list, 1)[0]
                sent = tmp_list[:char_index_to_substitute] + [shape_to_substitute] + tmp_list[(char_index_to_substitute + 1):]
                return ''.join(sent)
    return None