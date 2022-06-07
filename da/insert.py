import random
import synonyms

def insert_random_char(sent, delete_chars, pos_dict=None):
    # 万一您不想改善目前的情况的话 ， 我们会向法院告贵工厂，也要跟媒体说我们住民的困扰。
    tmp_list = list(sent)
    if len(tmp_list) > 2:
        char_index_to_insert = random.sample([i for i in range(len(tmp_list))], 1)[0]
        ran_char_to_insert = random.sample(delete_chars, 1)[0]
        incorrect_sent = tmp_list[:(char_index_to_insert)] + [ran_char_to_insert] + tmp_list[char_index_to_insert:]
        return ''.join(incorrect_sent)
    return None

def insert_random_word(sent, delete_words, pos_dict=None):
    # 万一您不想改善目前的情况的话 ， 我们会向法院告贵工厂，也要跟媒体说我们住民的困扰。
    tmp_list = list(sent)
    if len(tmp_list) > 2:
        char_index_to_insert = random.sample([i for i in range(len(tmp_list))], 1)[0]
        ran_word_to_insert = random.sample(delete_words, 1)[0]
        incorrect_sent = tmp_list[:char_index_to_insert] + [ran_word_to_insert] + tmp_list[char_index_to_insert:]
        return ''.join(incorrect_sent)
    return None

def insert_same_word(sent, pos_dict=None):
    # 万一您不想改善目前的情况的话 ， 我们会向法院告贵工厂，也要跟媒体说我们住民的困扰。
    sent = list(sent)
    if len(sent) > 2:
        char_index_to_insert = random.sample([i for i in range(len(sent))], 1)[0]
        incorrect_sent = sent[:char_index_to_insert] + [sent[char_index_to_insert]] + sent[char_index_to_insert:]
        return ''.join(incorrect_sent)
    else:
        return None

def insert_same_char(sent, pos_dict=None):
    # 万一您不想改善目前的情况的话 ， 我们会向法院告贵工厂，也要跟媒体说我们住民的困扰。
    tmp_list = list(sent)
    if len(tmp_list) > 2:
        char_index_to_insert = random.sample([i for i in range(len(tmp_list))], 1)[0]
        incorrect_sent = tmp_list[:char_index_to_insert] + [tmp_list[char_index_to_insert]] + tmp_list[char_index_to_insert:]
        return ''.join(incorrect_sent)
    return None

def insert_synonyms_word(sent, pos_dict=None):
    # 万一您不想改善目前的情况的话 ， 我们会向法院告贵工厂，也要跟媒体说我们住民的困扰。
    recusive_cnt = 0
    tmp_list = list(sent)
    while True and recusive_cnt <= 5:
        recusive_cnt += 1
        word_index_to_insert = random.sample([i for i in range(len(tmp_list))], 1)[0]
        word_to_insert = tmp_list[word_index_to_insert]
        synonyms_list = synonyms.nearby(word_to_insert, 10)[0]
        synonyms_score = synonyms.nearby(word_to_insert, 10)[1]
        synonyms_pass_index_list = [idx for idx, score in enumerate(synonyms_score)] #if score > 0.75 and score < 1.0]
        if len(synonyms_pass_index_list) > 1:
            synonyms_to_insert = synonyms_list[random.sample(synonyms_pass_index_list, 1)[0]]
            incorrect_sent = tmp_list[:word_index_to_insert] + [synonyms_to_insert] + tmp_list[word_index_to_insert:]
            return ''.join(incorrect_sent)

    return None
