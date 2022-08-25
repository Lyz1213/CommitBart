import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import random


class Bart_seq2seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, bart, config, args, beam_size=None, max_length=None, sos_id=None, eos_id=None, type = False):
        super(Bart_seq2seq, self).__init__()
        self.args = args
        self.bart = bart
        self.encoder = bart.encoder
        self.decoder = bart.decoder
        self.config = config
        self.register_buffer("final_logits_bias", torch.zeros((1, self.bart.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.bart.shared.num_embeddings, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        if type:
            self.token_type_embeddings = nn.Embedding(5, config.hidden_size)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()


    def forward(self, gen_input_ids = None, gen_tgt_ids = None, gen_type_ids = None, input_embeds = True):
        # outputs = self.encoder(source_ids, attention_mask=source_mask, token_type_ids = type_ids)
        # encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if gen_tgt_ids is not None:
            source_ids = gen_input_ids
            source_mask = gen_input_ids.ne(1)
            target_ids = gen_tgt_ids
            target_mask = gen_tgt_ids.ne(1)
            type_ids = gen_type_ids
            if input_embeds:
                token_embeds = self.encoder.embed_tokens(source_ids)
                type_embedding = self.token_type_embeddings(type_ids)
                inputs_embeds = token_embeds + type_embedding
                outs = self.bart(inputs_embeds=inputs_embeds, attention_mask=source_mask,
                                 decoder_input_ids=target_ids,
                                 decoder_attention_mask=target_mask)
            else:
                outs = self.bart(input_ids=source_ids, attention_mask=source_mask, decoder_input_ids=target_ids,
                                 decoder_attention_mask=target_mask)

            # hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(outs[0]) + self.final_logits_bias
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # source_ids = gen_input_ids
            # source_mask = gen_input_ids.ne(1)
            # if input_embeds:
            #     token_embeds = self.encoder.embed_tokens(source_ids)
            #     type_embedding = self.token_type_embeddings(gen_type_ids)
            #     inputs_embeds = token_embeds + type_embedding
            #     encoder_output = self.encoder(inputs_embeds=inputs_embeds,
            #                                   attention_mask=source_ids.ne(1))[0].permute([1, 0, 2]).contiguous()
            # else:
            #     encoder_output = self.encoder(input_ids=source_ids, attention_mask=source_mask)[0].permute(
            #         [1, 0, 2]).contiguous()
            # preds = []
            # zero = torch.cuda.LongTensor(1).fill_(0)
            # context = encoder_output
            # context_mask = source_mask
            # # input_ids = beam.getCurrentState()
            # input_ids = torch.zeros((context_mask.size(0),1)).long().to(self.args.device)
            # print('ori input_ids is ', input_ids.size())
            # print('context is ', context.size())
            # print('context mask is ', context_mask.size())
            # for _ in range(self.max_length):
            #     context = (context.permute([1, 0, 2]).contiguous())
            #     outs = self.decoder(input_ids=input_ids, encoder_hidden_states=context,
            #                         encoder_attention_mask=context_mask)
            #     out = self.lm_head(outs[0][:, -1, :]) + self.final_logits_bias
            #     out = self.lsm(out).data
            #     print('out size ', out.size())
            #     new_word = torch.argmax(out, dim = 1).unsqueeze(1)
            #     print('new word ', new_word[0])
            #     input_ids = torch.cat((input_ids, new_word), 1)
            #     print('inpuut_ids ', input_ids.size())
            #     print('input ', input_ids[0])
            #     # beam.advance(out)
            #     # input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            #     # input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            #     # print('input_ids is ', input_ids.size())
            #
            # preds = input_ids.unsqueeze(1)
            # print('preds size ', preds.size())
            # print('preds[0]', preds[0].size(), ' content: ', preds[0].size())
            # return preds
            source_ids = gen_input_ids
            source_mask = gen_input_ids.ne(1)
            if input_embeds:
                token_embeds = self.encoder.embed_tokens(source_ids)
                type_embedding = self.token_type_embeddings(gen_type_ids)
                inputs_embeds = token_embeds + type_embedding
                encoder_output = self.encoder(inputs_embeds=inputs_embeds,
                                              attention_mask=source_ids.ne(1))[0].permute([1, 0, 2]).contiguous()
            else:
                encoder_output = self.encoder(input_ids=source_ids, attention_mask=source_mask)[0].permute(
                    [1, 0, 2]).contiguous()
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    context = (context.permute([1, 0, 2]).contiguous())
                    outs = self.decoder(input_ids=input_ids, encoder_hidden_states=context,
                                        encoder_attention_mask=context_mask)
                    out = self.lm_head(outs[0][:, -1, :]) + self.final_logits_bias
                    out = self.lsm(out).data
                    #print('out size ', out.size())
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                    #print('input_ids is ', input_ids.size())
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                # print('pred ', len(pred))
                # print('pred[0] size ', pred[0])
                preds.append(torch.cat(pred, 0).unsqueeze(0))
                # print('preds[-1] size ', preds[-1].size())
                # print('preds len ', len(preds))
                # print('******************')
            preds = torch.cat(preds, 0)
            #print('preds size ', preds.size())
            return preds





class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
