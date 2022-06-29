#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import copy
import logging
import math
import os
import pkuseg

import numpy as np
import torch
import wandb
import operator
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from training.utils.seq2seq_utils import *
from torch.optim import SGD
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from evaluation.evaluate import Evaluator
from operator import itemgetter
# import bleurt 
#from evaluation.e_modules.tokenizer import Tokenizer as EvalTokenizer
#from evaluation.e_modules.annotator import Annotator as EvalAnnotator
#from evaluation.compare_m2_for_evaluation import calculate_score
import evaluation.m2score.levenshtein as levenshtein

class Seq2SeqTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None, tokenizer=None, preprocessor=None, test_edits_dict=None):
        self.args = args
        self.device = device

        # set data
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
        self.model.to(self.device)
        # self.tokenizer = tokenizer
        self.encoder_tokenizer = tokenizer[0]
        self.decoder_tokenizer = tokenizer[1]
        self.preprocessor = preprocessor

        # training results
        self.results = {}
        
        #eval_tokenizer = EvalTokenizer('word', self.device)
        #eval_annotator = EvalAnnotator.create_default('word', 'first')
        #self.evaluator = Evaluator(eval_tokenizer, eval_annotator)
        
        self.test_edits_dict = test_edits_dict
        self.max_unchanged_words=2
        self.beta = 0.5
        self.ignore_whitespace_casing= False
        self.verbose = False
        self.very_verbose = False
        self.seg = pkuseg.pkuseg()

    def set_data(self, train_dl, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def train_model(self, device=None):

        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        logging.info("args.gradient_accumulation_steps: " + str(self.args.gradient_accumulation_steps))
        self.model.to(device)

        args = self.args
        

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in self.model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in self.model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )
        
        iteration_in_total = len(
            self.train_dl) // args.gradient_accumulation_steps * args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)
        # warmup_steps = math.ceil(t_total * args.warmup_ratio)
        # args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
 

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        # train_iterator = trange(int(args.epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)

        for epoch in range(0, args.epochs):
            
            for batch_idx, batch in enumerate(self.train_dl):
                self.model.train()
                inputs = self._get_inputs_dict(batch)
                
                if args.fp16:
                    with amp.autocast():
                        outputs = self.model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = self.model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]
                
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                current_loss = loss.item()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx, len(self.train_dl), current_loss))

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                               and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _ = self.eval_model(epoch, global_step)
                        logging.info(results)
                        

        return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        f0_5_score = 0.0
        precision_score = 0.0
        recall_score = 0.0

        # bluert_score = 0.0
        # bluert_checkpoint = "~/fednlp_data/bleurt-base-128"
        # bleurt_scorer = bleurt.score.BleurtScorer(bluert_checkpoint)


        nb_eval_steps = 0

        n_batches = len(self.test_dl)

        test_sample_len = len(self.test_dl.dataset)
        # pad_token_label_id = self.pad_token_label_id
        eval_output_dir = self.args.output_dir

        preds = None
        out_label_ids = None

        self.model.to(device)
        self.model.eval()
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        logging.info('----- Evaluation Examples -----')
        for i, batch in enumerate(self.test_dl):
            # batch = tuple(t for t in batch)
            inputs = self._get_inputs_dict(batch)
            with torch.no_grad(): 
                outputs = self.model(**inputs)
                tmp_eval_loss = outputs[0]
                summary_ids = self.model.generate(inputs['input_ids'], num_beams=self.args.num_beams,
                                                  max_length=self.args.max_seq_length, early_stopping=True)
                wrong_tag_list = [''.join(self.decoder_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')).strip() for g in inputs['input_ids']]
                pred_tag_list = [''.join(self.decoder_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')).strip() for g in summary_ids]

                if i == 0:
                    logging.info('X: ' + wrong_tag_list[0])
                    #logging.info('Y: ' + gold_tag_list[0])
                    logging.info('P: ' + pred_tag_list[0])
                
                #hyp_input_sents, ref_input_sents = [], []
                #for j in range(len(wrong_tag_list)):
                #    hyp_input_sents.append([wrong_tag_list[j], pred_tag_list[j]])
                #    ref_input_sents.append([wrong_tag_list[j], gold_tag_list[j]])
                #hyp_annotations = self.evaluator.get_edits(hyp_input_sents, batch_size=self.args.train_batch_size)
                #ref_annotations = self.evaluator.get_edits(ref_input_sents, batch_size=self.args.train_batch_size)
                #result = calculate_score(hyp_annotations, ref_annotations)
                #f0_5_score += result['f0_5']
                #precision_score += result['precision']
                #recall_score += result['recall']
                
                system_sentences, source_sentences, gold_edits = [], [], []
                #logging.info(len(pred_tag_list))
                for num in range(len(pred_tag_list)):
                    sent = wrong_tag_list[num]
                    if sent in self.test_edits_dict:
                        #system_sentences = [' '.join(self.seg.cut(pred)) for pred in pred_tag_list]
                        system_sentences.append(' '.join(self.seg.cut(pred_tag_list[num])))
                        #source_sentences = [self.test_edits_dict[sent][0] for sent in wrong_tag_list]
                        source_sentences.append(self.test_edits_dict[sent][0])
                        #gold_edits = [self.test_edits_dict[sent][1] for sent in wrong_tag_list]
                        gold_edits.append(self.test_edits_dict[sent][1])
                    else:
                        logging.info('BIG WARNING FOR EVALUATION: Check ./training/ss_transformer_trainer.py line 296')
                        logging.info(sent)
                        logging.info('='*10)
                #logging.info(len(system_sentences))

                assert len(system_sentences) == len(source_sentences) == len(gold_edits)
                
                #for i in range(len(system_sentences)):
                #    logging.info(system_sentences[i])
                #    logging.info(source_sentences[i])
                #    logging.info('-'*20)

                p, r, f05 = levenshtein.batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits,
                                                              self.max_unchanged_words, self.beta, 
                                                              self.ignore_whitespace_casing, self.verbose, 
                                                              self.very_verbose)

                f0_5_score += f05
                precision_score += p
                recall_score += r
                
                # logits = output[0]
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += tmp_eval_loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))
            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            if i%100 == 0:
                logging.info("batch index = %d" % (i))

        eval_loss = eval_loss / nb_eval_steps
        f0_5_score = f0_5_score / nb_eval_steps
        precision_score = precision_score / nb_eval_steps
        recall_score = recall_score / nb_eval_steps

        result = {
            "eval_loss": eval_loss,
            "f0_5_score": f0_5_score,
            "recall_score": recall_score,
            "precision_score": precision_score
        }
        
        wandb.log(result)
        results.update(result)

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        self.results.update(result)

        model_preds = None
            
        logging.info(self.results)

        return result, model_preds

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        if self.args.fl_algorithm == "FedOPT" or self.args.fl_algorithm == "":
            optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        else:
            optimizer = SGD(model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

    def _get_inputs_dict(self, batch):
        device = self.device
        if self.args.model_type in ["bart", "marian"]:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": lm_labels.to(device),
            }
        elif self.args.model_type in ["mbart"]:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "decoder_input_ids": batch["decoder_input_ids"].to(device),
                "labels": batch["labels"].to(device),
            }
        elif self.args.model_type in ["bert_lm_zh", "roberta_lm_zh"]:
            lm_labels = batch[1]
            lm_labels_masked = lm_labels.clone()
            lm_labels_masked[lm_labels_masked == self.decoder_tokenizer.pad_token_id] = -100
            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": lm_labels.to(device),
                "labels": lm_labels_masked.to(device),
            } 
        elif self.args.model_type in ["bart_zh"]:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": lm_labels.to(device),
            }
        else:
            lm_labels = batch[1]
            lm_labels_masked = lm_labels.clone()
            lm_labels_masked[lm_labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": lm_labels.to(device),
                "labels": lm_labels_masked.to(device),
            }

        return inputs

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        # self._move_model_to_device()

        all_outputs = []
        # Batching
        for batch in [
            to_predict[i : i + self.args.eval_batch_size] for i in range(0, len(to_predict), self.args.eval_batch_size)
        ]:
            if self.args.model_type == "marian":
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            elif self.args.model_type in ["mbart"]:
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    src_texts=batch,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=True,
                    padding="max_length",
                    truncation=True,
                    src_lang=self.args.src_lang,
                )["input_ids"]
            else:
                input_ids = self.encoder_tokenizer.batch_encode_plus(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            input_ids = input_ids.to(self.device)

            if self.args.model_type in ["bart", "marian"]:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_seq_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            elif self.args.model_type in ["mbart"]:
                tgt_lang_token = self.decoder_tokenizer._convert_token_to_id(self.args.tgt_lang)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=tgt_lang_token,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_seq_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=self.model.config.decoder.pad_token_id,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_seq_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )

            all_outputs.extend(outputs.cpu().numpy())

        if self.args.use_multiprocessed_decoding:
            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=self.args.multiprocessing_chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.decoder_tokenizer.decode(
                    output_id, skip_special_tokens=self.args.skip_special_tokens, clean_up_tokenization_spaces=True
                )
                for output_id in all_outputs
            ]

        if self.args.num_return_sequences > 1:
            return [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            return outputs

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(
            output_id, skip_special_tokens=self.args.skip_special_tokens, clean_up_tokenization_spaces=True
        )
