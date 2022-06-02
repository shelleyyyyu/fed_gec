
def evaluate_gec(wrong_tag_list, gold_tag_list, pred_tag_list):
    print('-----'+'gec evaluation example'+'-----')
    print('wrong_tag_list', wrong_tag_list[0])
    print('gold_tag_list', gold_tag_list[0])
    print('pred_tag_list', pred_tag_list[0])
    right_true, right_false, wrong_true, wrong_false = 0, 0, 0, 0
    for glist, plist, wlist in zip(gold_tag_list, pred_tag_list, wrong_tag_list):
        for c, w, p in zip(glist, wlist, plist):
            if w == c:
                if p == c:
                    right_true += 1
                else:
                    right_false += 1
            else:
                if p == c:
                    wrong_true += 1
                else:
                    wrong_false += 1

    all_wrong = wrong_true + wrong_false
    recall = wrong_true / (all_wrong + 1e-8)
    precision = wrong_true / (right_false + wrong_true + + 1e-8)
    # F0.5-Measure = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    f0_5 = (1.25 * recall * precision) / (recall + (0.25 * precision)) + 1e-8
    return recall, precision, f0_5