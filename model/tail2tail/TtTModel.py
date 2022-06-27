import torch
from torch import nn
import torch.nn.functional as F
from crf_layer import DynamicCRF
from bert import BERTLM
from data import Vocab, CLS, SEP, MASK

def init_empty_bert_model(bert_args, bert_vocab, gpu_id):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, bert_args.approx)
    return bert_model

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters

def init_bert_model(args, device, bert_vocab):
    bert_ckpt= torch.load(args.bert_path)
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
        bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    if torch.cuda.is_available():
        bert_model = bert_model.cuda(device)
    if args.freeze == 1:
        for p in bert_model.parameters():
            p.requires_grad=False
    return bert_model, bert_vocab, bert_args

def ListsToTensor(xs, vocab):
    lens = [ len(x)+2 for x in xs]
    mx_len = max(lens)
    ys = []
    for i, x in enumerate(xs):
        y = vocab.token2idx([CLS]+x) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    data = torch.LongTensor(ys).t_().contiguous()
    return data

def batchify(data, vocab):
    return ListsToTensor(data, vocab)

class TtTModel(nn.Module):
    def __init__(self, bert_model, num_class, embedding_size, batch_size, dropout, device, vocab, args, loss_type='FC_FT_CRF'):
        super(TtTModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = dropout
        self.device = device
        self.batch_size = int(batch_size * (1 + args.augment_percentage))
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.vocab = vocab
        self.fc = nn.Linear(self.embedding_size, self.num_class)
        self.CRF_layer = DynamicCRF(num_class)
        self.loss_type = loss_type
        self.bert_vocab = vocab

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost), cost

    def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
        if gamma is None:
            gamma = 2
        p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1))
        g = (1 - torch.clamp(p, min=0.01, max=0.99)) ** gamma
        cost = -g * torch.log(p + 1e-8)
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost), g.view(y.shape), cost

    def forward(self, text_data, in_mask_matrix, in_tag_matrix, fine_tune=False, gamma=None, get_crf_detail=False):
        current_batch_size = len(text_data)
        max_len = 0
        for instance in text_data:
            max_len = max(len(instance), max_len)
        seq_len = max_len + 1  # 1 for [CLS]]

        # in_mask_matrix.size() == [batch_size, seq_len]
        # in_tag_matrix.size() == [batch_size, seq_len]
        mask_matrix = torch.tensor(in_mask_matrix, dtype=torch.uint8).t_().contiguous()
        tag_matrix = torch.LongTensor(in_tag_matrix).t_().contiguous()  # size = [seq_len, batch_size]
        if torch.cuda.is_available():
            mask_matrix = mask_matrix.cuda(self.device)
            tag_matrix = tag_matrix.cuda(self.device)
        assert mask_matrix.size() == tag_matrix.size()
        assert mask_matrix.size() == torch.Size([seq_len, current_batch_size])

        # input text_data.size() = [batch_size, seq_len]
        data = batchify(text_data, self.vocab)  # data.size() == [seq_len, batch_size]
        input_data = data
        if torch.cuda.is_available():
            data = data.cuda(self.device)

        sequence_representation = self.bert_model.work(data)[0]  # [seq_len, batch_size, embedding_size]
        if torch.cuda.is_available():
            sequence_representation = sequence_representation.cuda(self.device)  # [seq_len, batch_size, embedding_size]

        # dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)  # [seq_len, batch_size, embedding_size]
        sequence_representation = sequence_representation.view(current_batch_size * seq_len,
                                                               self.embedding_size)  # [seq_len * batch_size, embedding_size]
        sequence_emissions = self.fc(
            sequence_representation)  # [seq_len * batch_size, num_class]; num_class: 所有token in vocab
        sequence_emissions = sequence_emissions.view(seq_len, current_batch_size,
                                                     self.num_class)  # [seq_len, batch_size, num_class]; num_class: 所有token in vocab

        # bert finetune loss
        probs = torch.softmax(sequence_emissions, -1)
        if "FC" in self.loss_type:
            loss_ft_fc, g, loss_ft_fc_list = self.fc_nll_loss(probs, tag_matrix, mask_matrix, gamma=gamma)
        else:
            loss_ft, loss_ft_list = self.nll_loss(probs, tag_matrix, mask_matrix)

        sequence_emissions = sequence_emissions.transpose(0, 1)
        tag_matrix = tag_matrix.transpose(0, 1)
        mask_matrix = mask_matrix.transpose(0, 1)

        if "FC" in self.loss_type:
            loss_crf_fc, loss_crf_fc_list = self.CRF_layer(sequence_emissions, tag_matrix, mask=mask_matrix,
                                                           reduction='token_mean', g=None, gamma=gamma)
            loss_crf_fc = -1 * loss_crf_fc
            loss_crf_fc_list = -1 * loss_crf_fc_list
        else:
            loss_crf, loss_crf_list = self.CRF_layer(sequence_emissions, tag_matrix, mask=mask_matrix,
                                                     reduction='token_mean')
            loss_crf = -1 * loss_crf
            loss_crf_list = -1 * loss_crf_list

        decode_result = self.CRF_layer.decode(sequence_emissions, mask=mask_matrix, get_detail=get_crf_detail)
        self.best_decode_scores, self.best_decode_result, self.decode_result = decode_result
        self.best_decode_result = self.best_decode_result.tolist()

        if self.loss_type == 'CRF':
            loss_list = loss_crf_list.view(current_batch_size, -1)
            loss = loss_crf
            return self.best_decode_result, loss, loss_crf.item(), 0.0, input_data, loss_list, self.decode_result
        elif self.loss_type == 'FT_CRF':
            loss_list = loss_ft_list + loss_crf_list.view(current_batch_size, -1)
            loss = loss_ft + loss_crf
            return self.best_decode_result, loss, loss_crf.item(), loss_ft.item(), input_data, loss_list, self.decode_result
        elif self.loss_type == 'FC_FT_CRF':
            loss_list = loss_ft_fc_list + loss_crf_fc_list.view(current_batch_size, -1)
            loss = loss_ft_fc + loss_crf_fc
            return self.best_decode_result, loss, loss_crf_fc.item(), loss_ft_fc.item(), input_data, loss_list, self.decode_result
        elif self.loss_type == 'FC_CRF':
            loss_list = loss_crf_fc_list.view(current_batch_size, -1)
            loss = loss_crf_fc
            return self.best_decode_result, loss, loss_crf_fc.item(), 0.0, input_data, loss_list, self.decode_result
        else:
            print("error")
            return self.best_decode_result, 0, 0, 0, input_data, None, self.decode_result