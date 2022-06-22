import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--train_data',type=str)
    parser.add_argument('--dev_data',type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--label_data',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--dropout',type=float)
    parser.add_argument('--freeze',type=int)
    parser.add_argument('--number_class', type = int)
    parser.add_argument('--number_epoch', type = int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--bert_vocab', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--prediction_max_len', type=int)
    parser.add_argument('--dev_eval_path', type=str)
    parser.add_argument('--test_eval_path', type=str)
    parser.add_argument('--final_eval_path', type=str)
    parser.add_argument('--l2_lambda', type=float)
    parser.add_argument('--training_max_len', type=int)
    parser.add_argument('--restore_ckpt_path', type=str, default=None)

    # by_rule: random pos; rule-based;
    # by_pos_auto: unconfident pos; auto-based (based on model gen candidates)
    # by_pos_rule: rule-based; unconfident pos;

    parser.add_argument('--augment_percentage', type=float, default=1.0)
    parser.add_argument('--augment_method', default='by_rule', choices=['by_rule', 'by_pos_auto', 'by_pos_rule', None])
    parser.add_argument('--augment_cold_start_epoch', type=int, default=20)

    parser.add_argument('--augment_type', default='contributed', choices=['contributed', 'average', 'insert', 'delete', 'substitution', 'paraphrase'])

    parser.add_argument('--augment_descending', action='store_true')

    return parser.parse_args()