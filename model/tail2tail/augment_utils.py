from data import CLS, SEP, MASK, PAD


def get_candidates_dict_list(to_augment_correct_data_detail, bert_vocab):
    sent_candidate_groups = []
    for sent in to_augment_correct_data_detail:
        sent_list = []
        max_len = 0
        for words in sent:
            if len(words) > max_len:
                max_len = len(words)
        for words in sent:
            group = []
            for w in words:
                w = bert_vocab.idx2token(w)
                if w != CLS and w != MASK and w != SEP and w != PAD:
                    group.append(w)
            if len(group) > 0:
                sent_list.append(group)
        sent_candidate_groups.append(sent_list)

    candidates_dict_list = []
    for sent in sent_candidate_groups:
        have_candidates = [idx for idx, word_list in enumerate(sent) if len(word_list) > 1]
        candidates_dict = {}
        for i in have_candidates:
            to_aug_pos_index = i
            to_aug_pos_candidates = sent[to_aug_pos_index]
            candidates_dict[to_aug_pos_index] = to_aug_pos_candidates
        candidates_dict_list.append(candidates_dict)

    return candidates_dict_list