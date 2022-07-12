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
                if source in sentence_to_tokenized and target in sentence_to_tokenized:
                    source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
                    out, cors = self.annotator(source_tokenized, target_tokenized, idx)
                    if idx == 0:
                        output_str += "".join(out[:-1])
                    else:
                        output_str += "".join(out[1:-1])
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
