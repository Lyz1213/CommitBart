import gzip
import json
from transformers import PLBartTokenizer
import math
import torch
import re
import random
POS_change = '[POS]'
NEG_change = '[NEG]'
END_change = '[END]'
file_change = '[FILE]'
msg_change = '[MSG]'
code_change = '[CODE]'
msg_gen = '<MSG>'
pos_gen = '<POS>'
tif = '<TIF>'
not_mask = {'to','for', 'that','(',')', 'in','at','|','on','the','and','a', 'of'}
new_word = [POS_change, NEG_change, file_change, msg_change, code_change, END_change, msg_gen, pos_gen, tif]

class Data_Preprocessor:
    def __init__(self,
                 max_len,
                 target_max_len,
                 tokenizer,
                 model_type = 'roberta',
                 args = None,
                 mask_whole_word = False,
                 ):
        self.target_max_len = target_max_len
        self.model_type = model_type
        self.mask_whole_word = mask_whole_word
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(new_word)
        self.args = args
        self.url_regex = r"\"?http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\"?"
        self.unuse_regex = r"[-|+=_/#]{4,}"
        self.msg_id = self.tokenizer.convert_tokens_to_ids([msg_change])[0]
        self.file_id = self.tokenizer.convert_tokens_to_ids([file_change])[0]
        self.code_id = self.tokenizer.convert_tokens_to_ids([code_change])[0]

    def prefix_index(self, sample):
        ind = 0
        for i, c in enumerate(sample):
            if c == '+' or c=='-':
                ind += 1
            else:
                break
        return ind

    def tokenize_with_type(self, data, type = True ):
        data = re.sub(self.url_regex, 'URL', data)
        data = re.sub(self.unuse_regex, '', data)
        data = data.strip().replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace(
            '}', ' } ').replace('\t', '').replace('\n', '').replace('[POS] [END]', '').replace('[NEG] [END]', '').split()
        type_ids = []
        ids = []
        change_type = False
        code_part = False
        type_flag = 0
        for tok in data:
            if self.model_type != 'plbart':
                sub_tks = self.tokenizer.tokenize(tok, add_prefix_space=True)
            else:
                sub_tks = self.tokenizer.tokenize(tok)
            if len(sub_tks) == 0:
                continue
            ids += self.tokenizer.convert_tokens_to_ids(sub_tks)
            if type:
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
        if type:
            assert(len(type_ids) == len(ids))
        return ids, type_ids

    def pad_data(self, data, pad_token, max_len = 512):
        if len(data) > max_len:
            return None
        else:
            pad_len = max_len - len(data)
            data = data + [pad_token] * pad_len
        return data

    def shuffle(self, ids, type):
        print(ids)
        m_ind = (ids == self.msg_id).nonzero(as_tuple = True)[0]
        f_ind = (ids == self.file_id).nonzero(as_tuple = True)[0]
        c_ind = (ids == self.code_id).nonzero(as_tuple=True)[0]
        print('m_ind', m_ind, 'f_ind', f_ind, 'c_ind', c_ind)
        return ids, type

    def input2Features(self, sample):
        if self.args.finetune_task == 'pos':
            gen_input_ids, gen_tgt_ids, gen_type_ids = self.gen_token(sample, 'pos')
        else:
            gen_input_ids, gen_tgt_ids, gen_type_ids = self.gen_token(sample, 'msg')
        if gen_input_ids == None:
            return None, None, None
        assert(gen_input_ids.size(0) == self.max_len)
        assert (gen_tgt_ids.size(0) == self.target_max_len)
        assert (gen_type_ids.size() == gen_input_ids.size())
        return gen_input_ids, gen_tgt_ids, gen_type_ids


    def gen_token(self, sample, type):
        message = sample['summary']
        if message.strip() == '':
            return None, None, None
        diff_chunk = ''
        c_file = ''
        tgts_ = ''
        for diff in sample['diffs']:
            n_file = diff['negative_changed_file_name']
            p_file = diff['positive_changed_file_name']
            if n_file == p_file:
                c_file = file_change + ' ' + p_file
            else:
                c_file = file_change + ' - ' + n_file + ' + ' + p_file
            for chunk in diff['chunks']:
                ck = chunk['chunk_str']
                tgts__ = chunk['chunk_str']
                for i, p_change in enumerate(chunk['positive_changes']):
                    p_change = p_change.strip()
                    if p_change != '':
                        ind = self.prefix_index(p_change)
                        new_p = ' ' + POS_change + ' ' + p_change[ind:] + ' ' + END_change
                        if type == 'pos':
                            ck = ck.replace(p_change.strip(), '', 1)
                            tgts__ = tgts__.replace(p_change.strip(), new_p, 1)
                        else:
                            ck = ck.replace(p_change.strip(), new_p, 1)

                for i, n_change in enumerate(chunk['negative_changes']):
                    n_change = n_change.strip()
                    if n_change != '':
                        ind = self.prefix_index(n_change)
                        new_n = ' ' + NEG_change + ' ' + n_change[ind:] + ' ' + END_change
                        ck = ck.replace(n_change.strip(), new_n, 1)
                        if type == 'pos':
                            tgts__ = tgts__.replace(n_change.strip(), '')
                diff_chunk += ck + ' '
                tgts_ += tgts__ + ' '
        diff_chunk = code_change + ' ' + diff_chunk
        message = msg_change + ' ' + message

        f_ids, f_type = self.tokenize_with_type(c_file)
        m_ids, m_type = self.tokenize_with_type(message)
        d_ids, d_type = self.tokenize_with_type(diff_chunk)

        if type != 'msg':
            input_ids = self.tokenizer.convert_tokens_to_ids([pos_gen]) + m_ids + f_ids + d_ids
            inp_type_ids = [0] + m_type + f_type + d_type
            tgt_ids,_ = self.tokenize_with_type(tgts_, False)
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids([msg_gen]) + f_ids + d_ids
            inp_type_ids = [0] + f_type + d_type
            tgt_ids = m_ids

        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.eos_token_id]
        tgt_ids = [self.tokenizer.cls_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
        inp_type_ids = [0] + inp_type_ids + [0]
        input_ids = self.pad_data(input_ids, self.tokenizer.pad_token_id)
        tgt_ids = self.pad_data(tgt_ids, self.tokenizer.pad_token_id, self.target_max_len)
        inp_type_ids = self.pad_data(inp_type_ids, 0)
        if tgt_ids == None or inp_type_ids == None:
            return None, None, None
        return torch.tensor(input_ids).long(), torch.tensor(tgt_ids).long(), torch.tensor(inp_type_ids).long()



    def clear_data(self, data):
        data = re.sub(self.url_regex, 'URL', data)
        data = re.sub(self.unuse_regex, '', data)
        data = data.strip().replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace(
            '}', ' } ').replace('\t', '').replace('\n', '').split()
        return data

def load_jsonl_gz(file_name):
    instances = []
    with gzip.GzipFile(file_name, 'r') as f:
        lines = list(f)
    for i, line in enumerate(lines):
        instance = json.loads(line)
        instances.append(instance)
    return instances

def main():
    print('waiwaiwai?')
    data_path = '../../csharp_commits_valid.jsonl.gz'
    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
    print('pad id', tokenizer.pad_token_id)
    print('bos id ', tokenizer.bos_token_id, tokenizer.cls_token_id)
    print('eos id', tokenizer.eos_token_id, tokenizer.eos_token_id)
    print('vocab size ', tokenizer.vocab_size)
    tokenizer.add_tokens(new_word)
    for word in new_word:
        print('word ', word)
        print('tokenized', tokenizer.tokenize(word))
        print('id ', tokenizer.convert_tokens_to_ids(word))
    print('vocab size ', tokenizer.vocab_size)
    tokenizer.add_tokens(new_word)

    instances = load_jsonl_gz(data_path)
    print(instances[16])
    for i in range(100):
        pre = Data_Preprocessor(0.2, 512, tokenizer, without='msg')
        gen_input_ids, gen_tgt_ids, gen_type_ids, m_ids, d_ids, d_type, tf_input_ids, tf_labels, tf_type_ids = pre.input2Features(instances[i])
        print('i', i)
        print('gen_input ', gen_input_ids.size(), tokenizer.convert_ids_to_tokens(gen_input_ids))
        print('gen_tgt_ids ', gen_tgt_ids.size(),tokenizer.convert_ids_to_tokens(gen_tgt_ids))
        print('gen_type_ids ', gen_type_ids.size(),gen_type_ids)
        print('message ', m_ids.size(),tokenizer.convert_ids_to_tokens(m_ids))
        print('diff chunk ', d_ids.size(), tokenizer.convert_ids_to_tokens(d_ids))
        print('diff type ', d_type.size(), d_type)
        print('tf input ', tf_input_ids.size(),tokenizer.convert_ids_to_tokens(tf_input_ids))
        print('tf_labels ', tf_labels.size(),tokenizer.convert_ids_to_tokens(tf_labels))
        print('tf type ids ', tf_type_ids)

        print('***********************************\n\n\n')


if __name__ == "__main__":
    main()