import gzip
import json
from transformers import PLBartTokenizer
import math
import torch
import re
POS_change = '[POS]'
NEG_change = '[NEG]'
END_change = '[END]'
file_change = '[FILE]'
msg_change = '[MSG]'
code_change = '[CODE]'
msg_gen = '<MSG>'
pos_gen = '<POS>'
tif = '<TIF>'

new_word = [POS_change, NEG_change, file_change, msg_change, code_change, END_change, msg_gen, pos_gen, tif]

class Pos_Preprocessor:
    def __init__(self,
                 source_max_len,
                 target_max_len,
                 tokenizer,
                 model_type
                 ):
        self.src_max_len = source_max_len
        self.tgt_max_len = target_max_len
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(new_word)
        self.model_type = model_type
    def prefix_index(self, sample, p):
        ind = 0
        for i, c in enumerate(sample):
            if c == '+' or c=='-':
                ind += 1
            else:
                break
        return ind
    def input2Features(self, sample):
        input_ids = []
        inp_ = sample['inp']
        tgts_ = sample['pos']
        #print('inp_ ', inp_)
        #print('tgts_', tgts_)
        type_ids = []
        change_type = False
        code_part = False
        type_flag = 0
        toks = []
        # print('inp_ is ', inp_)
        tgts = []
        tgt_ids = []
        for tok in tgts_:
            #print('model type is ', self.model_type)
            if self.model_type != 'plbart':
                sub_tks = self.tokenizer.tokenize(tok, add_prefix_space=True)
            else:
                sub_tks = self.tokenizer.tokenize(tok)
            if len(sub_tks) == 0:
                continue
            tgts += sub_tks
            tgt_ids += self.tokenizer.convert_tokens_to_ids(sub_tks)
        tgt_ids = [self.tokenizer.cls_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
        if len(tgt_ids) > self.tgt_max_len:
            #print('return none since tgt out of max len, tgt len is ', len(tgt_ids))
            return None, None, None
        else:
            tgt_pad = self.tgt_max_len - len(tgt_ids)
            tgt_ids += [self.tokenizer.pad_token_id] * tgt_pad
        for tok in inp_:
            if self.model_type != 'plbart':
                sub_tks = self.tokenizer.tokenize(tok, add_prefix_space=True)
            else:
                sub_tks = self.tokenizer.tokenize(tok)
            if len(sub_tks) == 0:
                continue
            toks += sub_tks
            input_ids += self.tokenizer.convert_tokens_to_ids(sub_tks)
            if change_type:
                change_type = False
                type_flag = 4 if code_part else 0
            if tok == POS_change:
                type_flag = 1
            elif tok == NEG_change:
                type_flag = 2
            elif tok == file_change:
                type_flag = 3
            elif tok == code_change:
                type_flag = 4
                code_part = True
            elif tok == END_change:
                change_type = True
            type_ids += [type_flag] * len(sub_tks)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.eos_token_id]
        type_ids = [0] + type_ids + [0]
        assert len(type_ids) == len(input_ids)
        if len(input_ids) > self.src_max_len:
            #print('return none since inp out of max len, inp len is ', len(input_ids))
            return None, None, None
        else:
            pad_len = self.src_max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            type_ids += [0] * pad_len
        input_ids = torch.tensor(input_ids)
        # print('sample is ', sample)
        # print('inp ', self.tokenizer.convert_ids_to_tokens(input_ids))
        # print('tgt ', self.tokenizer.convert_ids_to_tokens(tgt_ids))
        return input_ids.long(), torch.tensor(tgt_ids).long(), torch.tensor(type_ids).long()

def load_jsonl_gz(file_name):
    instances = []
    with gzip.GzipFile(file_name, 'r') as f:
        lines = list(f)
    for i, line in enumerate(lines):
        instance = json.loads(line)
        instances.append(instance)
    return instances

def main():
    data_path = '../../python_commits_test.jsonl.gz'
    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
    print('vocab size ', tokenizer.vocab_size)
    tokenizer.add_tokens(new_word)
    for word in new_word:
        print('word ', word)
        print('tokenized', tokenizer.tokenize(word))
        print('id ', tokenizer.convert_tokens_to_ids(word))

    instances = load_jsonl_gz(data_path)
    print(instances[12])
    pre = Gen_Preprocessor(512, 215, tokenizer, 'msg')
    inp_ids, tgt_ids, type_ids = pre.input2Features(instances[12])



if __name__ == "__main__":
    main()