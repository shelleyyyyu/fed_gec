import random

def exchange_word(sent, pos_dict=None):
    # 万一 您 不想 改善目前 的 情况 的话 ， 我们 会 向 法院 告贵 工厂 ， 也 要 跟 媒体 说 我们 住民 的 困扰 。
    tmp_list = list(sent)
    if len(tmp_list) > 2:
        word_index_to_exchange = random.sample([i for i in range(len(tmp_list)-2)], 1)[0]
        tmp_str = tmp_list[word_index_to_exchange]
        tmp_list[word_index_to_exchange] = tmp_list[word_index_to_exchange+1]
        tmp_list[word_index_to_exchange + 1] = tmp_str
        return ''.join(tmp_list)
    return None


def exchange_char(sent, pos_dict=None):
    tmp_list = list(sent)
    if len(tmp_list) > 2:
        word_index_to_exchange = random.sample([i for i in range(len(tmp_list)-2)], 1)[0]
        tmp_str = tmp_list[word_index_to_exchange]
        tmp_list[word_index_to_exchange] = tmp_list[word_index_to_exchange+1]
        tmp_list[word_index_to_exchange + 1] = tmp_str
        return ''.join(tmp_list)
    return None