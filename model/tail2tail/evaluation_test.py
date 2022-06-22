import torch
from data_loader import DataLoader
from funcs import *
from TtTModel import TtTModel, extract_parameters, init_bert_model, init_empty_bert_model
from arguments import parse_config

if __name__ == "__main__":
    args = parse_config()
    print('Initializing model...')

    bert_args, model_args, bert_vocab, model_parameters = extract_parameters(args.restore_ckpt_path)
    bert_model = init_empty_bert_model(bert_args, bert_vocab, args.gpu_id)

    id_label_dict = {}
    label_id_dict = {}
    for lid, label in enumerate(bert_vocab._idx2token):
        id_label_dict[lid] = label
        label_id_dict[label] = lid

    batch_size = args.batch_size
    number_class = len(id_label_dict)  # args.number_class
    embedding_size = bert_args.embed_dim
    fine_tune = args.fine_tune
    loss_type = args.loss_type
    l2_lambda = args.l2_lambda
    model = TtTModel(bert_model, number_class, embedding_size, batch_size, args.dropout, args.gpu_id, bert_vocab, args, loss_type)
    model.load_state_dict(model_parameters)
    if torch.cuda.is_available():
        model = model.cuda(args.gpu_id)

    print('Model construction finished.')

    # Data Preparation
    train_path, test_path, test_path = args.train_data, args.test_data, args.test_data
    # label_path = args.label_data
    train_max_len = args.training_max_len
    nerdata = DataLoader(train_path, test_path, test_path, bert_vocab, train_max_len)
    print('data is ready')

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # --- training part ---#
    num_epochs = args.number_epoch
    training_data_num, test_data_num, test_data_num = nerdata.train_num, nerdata.test_num, nerdata.test_num
    test_step_num = test_data_num
    test_eval_path = args.test_eval_path

    model.eval()
    gold_tag_list, wrong_tag_list, pred_tag_list = [], [], []
    with torch.no_grad():
        with open(test_eval_path, 'w', encoding='utf8') as o:
            for test_step in range(test_data_num):
                test_batch_text_list, test_batch_tag_list, _ = nerdata.get_next_batch(batch_size=1, mode='test')
                test_tag_matrix = process_batch_tag(test_batch_tag_list, nerdata.label_dict)
                test_mask_matrix = make_mask(test_batch_tag_list)
                test_batch_result, _, _, _, test_input_data, _, _ = model(test_batch_text_list, test_mask_matrix, test_tag_matrix, fine_tune=False)

                test_text = ''
                for token in test_batch_text_list[0]:
                    test_text += token + ' '
                test_text = test_text.strip()

                # GroundTruth
                ground_text = ''
                for token in test_batch_tag_list[0]:
                    ground_text += id_label_dict[token] + ' '
                ground_text = ground_text.strip()

                valid_test_text_len = len(test_batch_text_list[0])
                test_tag_str = ''
                pred_tags = []
                for tag in test_batch_result[0][1:valid_test_text_len + 1]:
                    test_tag_str += id_label_dict[int(tag)] + ' '
                    pred_tags.append(int(tag))
                test_tag_str = test_tag_str.strip()

                o.writelines(test_text + '\t' + test_tag_str + '\t' + ground_text + '\n')
                wrong_tag_list.append(test_input_data[1:].t()[0].tolist())
                gold_tag_list.append(test_batch_tag_list[0])
                pred_tag_list.append(pred_tags)

            assert len(gold_tag_list) == len(pred_tag_list)
            right_true, right_false, wrong_true, wrong_false = 0, 0, 0, 0
            all_right, all_wrong = 0, 0

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
            far = right_false / (right_true + right_false)
            recall_wrong = wrong_true + wrong_false
            recall = wrong_true / all_wrong
            precision = wrong_true / (right_false + wrong_true)
            # F-Measure = (2 * Precision * Recall) / (Precision + Recall)
            f1 = (2 * recall * precision) / (recall + precision + 1e-8)
            # F0.5-Measure = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
            f0_5 = (1.25 * recall * precision) / (recall + (0.25 * precision)) + 1e-8
            print('Official f0.5 : %.4f, precision : %.4f, recall : %.4f, far_wrong : %.4f' % (f0_5, precision, recall, far))
