

import torch
import logging
import json

from torch import nn
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartConfig
from .model_bart_layer import BartConstrainedGen
from .common_utils import load_ontology, extract_args_from_template
from collections import defaultdict

logger = logging.getLogger(__name__)

sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
MAX_LENGTH = 512


class DocLvlEventExt_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

        self.config = BartConfig.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>', ' <tgr>'])

        if self.hparams.model == 'gen':
            self.model = BartGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings()
        elif self.hparams.model == 'constrained-gen':
            self.model = BartConstrainedGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings()
        else:
            raise NotImplementedError

        self.pair_constraints = {
            ('Justice.Sentence.Unspecified_JudgeCourt',
             'Life.Die.Unspecified_Victim'),
            ('Justice.Sentence.Unspecified_Defendant',
             'Life.Die.Unspecified_Victim'),
            ('Control.ImpedeInterfereWith.Unspecified_Impeder',
             'Justice.ArrestJailDetain.Unspecified_Jailer'),
            ('Contact.RequestCommand.Unspecified_Recipient',
             'Justice.ArrestJailDetain.Unspecified_Jailer'),
            ('Life.Injure.Unspecified_Injurer',
             'Transaction.ExchangeBuySell.Unspecified_Giver'),
            ('Justice.TrialHearing.Unspecified_Defendant',
             'Transaction.ExchangeBuySell.Unspecified_Giver'),
            ('Justice.TrialHearing.Unspecified_Defendant',
             'Transaction.ExchangeBuySell.Unspecified_Recipient'),
            ('Conflict.Attack.DetonateExplode_Attacker',
             'Contact.Contact.Broadcast_Communicator'),
            ('Conflict.Attack.Unspecified_Attacker',
             'Contact.Contact.Broadcast_Communicator'),
            ('Conflict.Attack.DetonateExplode_Attacker',
             'Contact.ThreatenCoerce.Unspecified_Communicator'),
            ('Conflict.Attack.Unspecified_Attacker',
             'Contact.ThreatenCoerce.Unspecified_Communicator'),
        }

        self.up_constrains = {
            "Killer_Attacker_Injurer_Damager_Destroyer": "Killer_Attacker_Destroyer_Defendant",
            "JudgeCourt": "JudgeCourt",
        }
        self.up_thresh = 4

        self.ontology_dict = load_ontology(dataset="KAIROS")
        for key in self.ontology_dict:
            for role in self.ontology_dict[key]['arg_to_prev']:
                w = self.ontology_dict[key]['arg_to_prev'][role]
                if w == '<s>':
                    self.ontology_dict[key]['arg_to_prev'][role] = [
                        w, 2]  # </s> decoder_start_token
                else:
                    w_list = self.tokenizer.tokenize(w, add_prefix_space=True)
                    self.ontology_dict[key]['arg_to_prev'][role] = [w, self.tokenizer.encode_plus(
                        w_list, add_special_tokens=True, add_prefix_space=True)['input_ids'][-2]]

        # import ipdb; ipdb.set_trace()

        self.memory = {}
        self.memory_down = {}
        self.memory_up_cnt = defaultdict(int)

        with open("preprocessed_KAIROS/test.jsonl", 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                doc_key = ex["doc_key"]
                evt_type = ex['event_type']
                if doc_key not in self.memory:
                    self.memory[doc_key] = {}
                    self.memory_down[doc_key] = {}
                    self.memory_up_cnt[doc_key] = {}
                # if evt_type not in self.memory[doc_key]:
                    # down
                    for evt_type in self.ontology_dict:
                        self.memory[doc_key][evt_type] = {}
                        self.memory_down[doc_key][evt_type] = {}
                        for role in self.ontology_dict[evt_type]['roles']:
                            if role not in self.memory[doc_key][evt_type]:
                                self.memory[doc_key][evt_type][role] = []
                                self.memory_down[doc_key][evt_type][role] = []
                    # up
                    for role_grp_key, role_grp in self.up_constrains.items():
                        if role_grp not in self.memory_up_cnt[doc_key]:
                            # ent1: #, ent2: #
                            self.memory_up_cnt[doc_key][role_grp] = {}

                            if role_grp_key == 'JudgeCourt':
                                ent = "George O'Toole Jr."
                                # import ipdb; ipdb.set_trace()
                                w_list = self.tokenizer.tokenize(
                                    "Jr.", add_prefix_space=True)
                                out_id = self.tokenizer.encode_plus(
                                    w_list, add_special_tokens=True, add_prefix_space=True)['input_ids'][1]
                                self.memory_up_cnt[doc_key][role_grp][ent] = [
                                    out_id, self.up_thresh]

        self.all_output_templates, self.all_out_template_embs = {}, {}
        for doc_key in self.memory:
            if doc_key not in self.all_output_templates:
                self.all_output_templates[doc_key] = []
                self.all_out_template_embs[doc_key] = []

    def forward(self, inputs):

        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_token_ids"],
            "attention_mask": batch["input_attn_mask"],
            "decoder_input_ids": batch['tgt_token_ids'],
            "decoder_attention_mask": batch["tgt_attn_mask"],
            "task": 0
        }

        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        log = {
            'train/loss': loss,
        }
        return {
            'loss': loss,
            'log': log
        }

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_token_ids"],
            "attention_mask": batch["input_attn_mask"],
            "decoder_input_ids": batch['tgt_token_ids'],
            "decoder_attention_mask": batch["tgt_attn_mask"],
            "task": 0,
        }
        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val/loss': avg_loss,
        }
        return {
            'loss': avg_loss,
            'log': log
        }

    def test_step(self, batch, batch_idx):

        if self.hparams.knowledge_pair_gen:
            doc_key = batch['doc_key'][-1]
            evt_type = batch['event_type'][-1]
            id_pairs_down = {}
            id_pairs_down_print = {}

            for role, ents in self.memory_down[doc_key][evt_type].items():
                in_id = self.ontology_dict[evt_type]['arg_to_prev'][role][-1]
                if ents:
                    down_out_ids = []
                    down_out_ids_print = []
                    for ent in ents[:]:  # for ent in ents: limited mem
                        down_out_ids.append(ent[-1])
                        down_out_ids_print.append(ent[:-1])
                    # import ipdb; ipdb.set_trace()
                    id_pairs_down[in_id] = down_out_ids
                    id_pairs_down_print[self.ontology_dict[evt_type]
                                        ['arg_to_prev'][role][0]] = down_out_ids_print

                    # fix participant exception (2 roles)
                    if role == "Participant":
                        in_id2 = 19  # " with"
                        id_pairs_down[in_id2] = down_out_ids
                        # import ipdb; ipdb.set_trace()
            id_pairs_up = {}
            for role in self.ontology_dict[evt_type]['roles']:
                for role_grp_key, role_grp in self.up_constrains.items():
                    if role in role_grp:
                        in_id = self.ontology_dict[evt_type]['arg_to_prev'][role][-1]
                        for ent in self.memory_up_cnt[doc_key][role_grp]:
                            if self.memory_up_cnt[doc_key][role_grp][ent][-1] >= self.up_thresh and self.memory_up_cnt[doc_key][role_grp][ent][0] in batch['input_token_ids']:
                                if in_id not in id_pairs_up:
                                    id_pairs_up[in_id] = []
                                id_pairs_up[in_id].append(
                                    self.memory_up_cnt[doc_key][role_grp][ent][0])

        # # import ipdb; ipdb.set_trace()
        input_token_ids = batch['input_token_ids']
        if self.hparams.sim_train:
            # calculate sbert embedding and find/add most similar
            doc_key = batch['doc_key'][0]
            context_emb = sim_model.encode(
                batch['context_words'][0], show_progress_bar=False)
            most_sim_out_template = []
            context = batch['context_tokens'][0]
            if len(self.all_out_template_embs[doc_key]) > 0:
                cosine_scores = util.pytorch_cos_sim(
                    [context_emb], self.all_out_template_embs[doc_key])
                most_sim_idx = torch.argmax(cosine_scores, dim=-1)
                # if len(all_out_template_embs[doc_key])>2: import ipdb; ipdb.set_trace()
                most_sim_out_template = self.all_output_templates[doc_key][most_sim_idx]
            context = most_sim_out_template+['</s>']+context
            input_tokens = self.tokenizer.encode_plus(batch['input_template'][0], context,
                                                      add_special_tokens=True,
                                                      add_prefix_space=True,
                                                      max_length=MAX_LENGTH,
                                                      truncation='only_second',
                                                      padding='max_length')
            input_token_ids = torch.stack(
                [torch.LongTensor(input_tokens['input_ids'])])
            if batch['input_token_ids'].device.type != 'cpu':
                input_token_ids = input_token_ids.cuda()

        # gen without id_pairs
        sample_output_no_knowledge = self.model.generate({}, {}, input_token_ids, do_sample=False,  # batch['input_token_ids']
                                                         max_length=30, num_return_sequences=1, num_beams=1,)

        if self.hparams.knowledge_pair_gen:
            if self.hparams.sample_gen:
                sample_output = self.model.generate(input_token_ids, do_sample=True,
                                                    top_k=20, top_p=0.95, max_length=30, num_return_sequences=1, num_beams=1,
                                                    )
            else:
                # id_pairs_down, id_pairs_up = {}, {}
                sample_output = self.model.generate(id_pairs_down, id_pairs_up, input_token_ids, do_sample=False,
                                                    max_length=30, num_return_sequences=1, num_beams=1,
                                                    )

            # add into memory
            doc_key = batch['doc_key'][-1]
            evt_type = batch['event_type'][-1]
            pred_template = self.tokenizer.decode(
                sample_output.squeeze(0), skip_special_tokens=True)
            predicted_args = extract_args_from_template(
                self.ontology_dict, evt_type, pred_template)
            # memory_place_cache = []
            for role in predicted_args:
                for ent in predicted_args[role]:
                    if not ent:
                        continue
                    w_list = self.tokenizer.tokenize(
                        ent[0], add_prefix_space=True)
                    out_id = self.tokenizer.encode_plus(
                        w_list, add_special_tokens=True, add_prefix_space=True)['input_ids'][1]
                    ent.append(out_id)
                    self.memory[doc_key][evt_type][role].append(ent)
                    # down
                    evt_type_role = "_".join([evt_type, role])
                    for pair in self.pair_constraints:
                        if evt_type_role == pair[0]:
                            evt_type2, role2 = pair[1].split("_")
                            self.memory_down[doc_key][evt_type2][role2].append(
                                ent)

                        if evt_type_role == pair[1]:
                            evt_type2, role2 = pair[0].split("_")
                            self.memory_down[doc_key][evt_type2][role2].append(
                                ent)
                    # up
                    for role_grp_key, role_grp in self.up_constrains.items():
                        if role in role_grp_key:
                            if ent[0] not in self.memory_up_cnt[doc_key][role_grp]:
                                self.memory_up_cnt[doc_key][role_grp][ent[0]] = [
                                    out_id, 1]
                            else:
                                self.memory_up_cnt[doc_key][role_grp][ent[0]][-1] += 1

            if id_pairs_down:  # if id_pairs_up:
                # print(id_pairs_up)
                # if batch_idx == 150: import ipdb; ipdb.set_trace()
                print("ored:", self.tokenizer.decode(
                    sample_output_no_knowledge.squeeze(0), skip_special_tokens=True))
                print("pred:", pred_template)
                print("gold:", self.tokenizer.decode(
                    batch['tgt_token_ids'][0], skip_special_tokens=True))
            else:
                sample_output = sample_output_no_knowledge

        sample_output = sample_output.reshape(
            batch['input_token_ids'].size(0), 1, -1)
        doc_key = batch['doc_key']  # list
        tgt_token_ids = batch['tgt_token_ids']

        if self.hparams.sim_train:
            # add new output_template
            output_template = self.tokenizer.decode(
                sample_output[0][0], skip_special_tokens=True)
            # import ipdb; ipdb.set_trace()
            out_template_emb = sim_model.encode(
                output_template, show_progress_bar=False)

            space_tokenized_template = output_template.split()
            tokenized_output_template = []
            for w in space_tokenized_template:
                tokenized_output_template.extend(
                    self.tokenizer.tokenize(w, add_prefix_space=True))

            self.all_output_templates[doc_key[0]].append(
                tokenized_output_template)
            self.all_out_template_embs[doc_key[0]].append(out_template_emb)

        return (doc_key, sample_output, tgt_token_ids)

    def test_epoch_end(self, outputs):
        # evaluate F1
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name), 'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):

                    pred = {
                        'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True)
                    }
                    writer.write(json.dumps(pred)+'\n')

        return {}
