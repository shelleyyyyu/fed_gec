import os
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Evaluator:
    
    def __init__(self, tokenizer, annotator):
        self.tokenizer = tokenizer
        self.annotator = annotator

    def annotate(self, sent_list, sentence_to_tokenized):
        """
        :param line:
        :return:
        """
        source = sent_list[0]
        source = "".join(source.strip().split())
        output_str = ""
        for idx, target in enumerate(sent_list[1:]):
            try:
                target = "".join(target.strip().split())
                #logging.info('checkcheckcheckcheckcheckcheckcheck')
                if source in sentence_to_tokenized and target in sentence_to_tokenized:
                    #logging.info('inininininininininin')
                    source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
                    #logging.info(source_tokenized)
                    #logging.info(target_tokenized)
                    out, cors = self.annotator(source_tokenized, target_tokenized, idx)
                    if idx == 0:
                        output_str += "".join(out[:-1])
                    else:
                        output_str += "".join(out[1:-1])
                #else:
                #    logging.info('outoutoutoutoutoutoutoutout')
                #    logging.info(source)
                #    logging.info(target)
            except Exception:
                raise Exception
        return output_str


    def get_edits(self, input_sents, batch_size=1):
        lines = input_sents
        annotations = []
        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}

        for sent_list in lines:
            for idx, sent in enumerate(sent_list):
                sentence_set.add(''.join(sent.split(' ')).strip())
        batch = []

        for sent in sentence_set:
            count += 1
            if sent:
                batch.append(sent)
            if count % batch_size == 0:
                results = self.tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.
                batch = []
        if batch:
            results = self.tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
                
        for line in lines:
            ret = self.annotate(line, sentence_to_tokenized)
            annotations.append(ret.strip())

        return annotations


# hyp_input_sents = [['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你就会尝到泰国人死爱的味道。']]
# ref_input_sents = [['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。'], ['这样，你就会尝到泰国人死爱的味道。', '这样，你会尝到泰国人死爱的味道。']]
#
# # granularity = 'char' # 'word'
# # multi_cheapest_strategy = "first" #, "all"
# # batch_size = 128 dev_batch_size
# # device = 0
#
# hyp_annotations = get_edits(hyp_input_sents, batch_size=128)
# ref_annotations = get_edits(ref_input_sents, batch_size=128)
# result = calculate_score(hyp_annotations, ref_annotations)
# print(result)