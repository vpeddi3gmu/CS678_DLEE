import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    BartModel,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation_utils import top_k_top_p_filtering
from typing import Iterable, List, Optional
from transformers.file_utils import ModelOutput


class BartConstrainedGen(PreTrainedModel):
    def __init__(self, config, tokenizer):
        super(BartConstrainedGen, self).__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.transformer = BartModel.from_pretrained('facebook/bart-large')
        # here we can keep config.vocab_size instead of self.transformer.shared.num_embeddings
        self.register_buffer("final_logits_bias",
                             torch.zeros((1, config.vocab_size)))
        # print(self.transformer.shared) #gives nn.embedding layer Embedding(50265, 1024, padding_idx=1)
        # print(self.transformer.shared.num_embeddings) # vocabsize which is 50265
        # print(config.vocab_size) # vocabsize which is 50265
        # print(config.d_model) #embedding size which is 1024
        # print(self.transformer.shared.weight) and print(self.transformer.shared.weight.data) #Parameter containing:
# tensor([[-0.0370,  0.1117,  0.1829,  ...,  0.2054,  0.0578, -0.0750],
#         [ 0.0055, -0.0049, -0.0069,  ..., -0.0030,  0.0038,  0.0087],
#         [-0.0448,  0.4604, -0.0604,  ...,  0.1073,  0.0310,  0.0477],
#         ...,
#         [-0.0138,  0.0278, -0.0467,  ...,  0.0455, -0.0265,  0.0125],
#         [-0.0043,  0.0153, -0.0567,  ...,  0.0496,  0.0108, -0.0099],
#         [ 0.0053,  0.0324, -0.0179,  ..., -0.0085,  0.0223, -0.0020]],
#        requires_grad=True)
 #  print(self.transformer.shared.weight.shape) #torch.Size([50265, 1024])

    def resize_token_embeddings(self):
        old_num_tokens = self.config.vocab_size  # actual vocabulary
        new_embeddings = self.transformer.resize_token_embeddings(
            len(self.tokenizer))
       # print(self.transformer.resize_token_embeddings(len(self.tokenizer))) # Embedding(50267, 1024)
       # print(len(self.tokenizer)) # 50267
        self.transformer.shared = new_embeddings
        self._resize_final_logits_bias(len(self.tokenizer), old_num_tokens)
        self.vocab_size = len(self.tokenizer)
        return new_embeddings

    # we can use config.vocab_size instead of old_num_tokens and change function defination as well
    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            # gives 1st len(new_num_tokens) from final logits bias, so can change this
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_encoder(self):
        return self.transformer.encoder

    def prepare_inputs_for_generation(
            self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, inputs_embeds, encoder_input_ids, **kwargs):
        return {
            "input_ids": encoder_input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            # change this to avoid caching (presumably for debugging)
            "use_cache": use_cache,
            "inputs_embeds": inputs_embeds,
        }

    def convert_pointer_logits_to_lm_logits(self, pointer_logits, input_ids):
        '''
        pointer_logits: (batch, seq_len, input_seq_len)
        input_ids: (batch, input_seq_len)
        lm_logits: (batch, seq_len, vocab_size)
        '''
        batch_size = pointer_logits.size(0)
        seq_len = pointer_logits.size(1)
        input_seq_len = input_ids.size(1)
        lm_logits = torch.full((batch_size, seq_len, self.vocab_size), fill_value=-
                               1000, dtype=pointer_logits.dtype).to(pointer_logits.device)

        #  scatter may be technically incorrect for duplicate indexes, but not using it gets slow
        index = input_ids.unsqueeze(dim=1).expand_as(pointer_logits)
        lm_logits.scatter_(dim=2, index=index, src=pointer_logits)

        return lm_logits

    def forward(self, input_ids,
                attention_mask=None,
                encoder_outputs=None,
                use_cache=False,
                past_key_values=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                inputs_embeds=None,
                task=-1):
        # import ipdb; ipdb.set_trace()
        # generation
        if task == -1:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=use_cache,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                inputs_embeds=inputs_embeds,
                return_dict=return_dict,)

            # print(outputs)
            # print( self.transformer.encoder.embed_scale) #1.0

            decoder_output = outputs[0]  # (batch, seq_len, hidden_dim)
            if encoder_outputs == None:
                # (batch, input_seq_len, hidden_dim)
                encoder_outputs = outputs[1]
                # BaseModelOutput if return dict

            if inputs_embeds == None:
                # get encoder side embeddings
                inputs_embeds = self.transformer.encoder.embed_tokens(
                    input_ids) * self.transformer.encoder.embed_scale  # (batch, seq_len, input_seq_len)
            # (batch, seq_len, input_seq_len)
            pointer_logits = torch.einsum(
                'ijk,ilk->ijl', decoder_output, inputs_embeds)
            lm_logits = self.convert_pointer_logits_to_lm_logits(
                pointer_logits, input_ids)

            masked_lm_loss = None

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        # training
        elif task == 0:

            assert(decoder_input_ids != None)
            y_ids = decoder_input_ids[:, :-1]
            labels = decoder_input_ids[:, 1:].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            # labels are just decoder_input_ids shifted to the right by 1

            outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=y_ids,
                decoder_attention_mask=decoder_attention_mask[:, :-1],
                use_cache=False,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)

            decoder_output = outputs[0]  # (batch, seq_len, hidden_dim)
            encoder_output = outputs[1]  # (batch, input_seq_len, hidden_dim)
            # lm_logits = F.linear(decoder_output, self.transformer.shared.weight, bias=self.final_logits_bias)
            # lm_logits = self.remove_unseen(lm_logits, input_ids)
            # get encoder side embeddings
            inputs_embeds = self.transformer.encoder.embed_tokens(
                input_ids) * self.transformer.encoder.embed_scale  # (batch, seq_len, input_seq_len)

            # (batch, seq_len, input_seq_len)
            pointer_logits = torch.einsum(
                'ijk,ilk->ijl', decoder_output, inputs_embeds)
            # decrease <arg> prob if neccesary

            lm_logits = self.convert_pointer_logits_to_lm_logits(
                pointer_logits, input_ids)

            # Add cache, hidden states and attention if they are here
            outputs = (lm_logits,) + outputs[1:]
            loss_fct = nn.CrossEntropyLoss()

            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

            return outputs

    # # this is a simplified generate class for the pointer generator taken from https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/generation_utils.py

    @torch.no_grad()
    def generate(
        self,
        id_pairs_down: Optional[dict] = None,
        id_pairs_up: Optional[dict] = None,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            # overriden by the input batch_size
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1

        assert isinstance(
            max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(
            min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(
            early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(
            num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(
            top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(
                bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(
                self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(
                self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(
                input_ids, attention_mask=attention_mask, return_dict=True)
            inputs_embeds = encoder.embed_tokens(
                input_ids) * encoder.embed_scale

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        encoder_input_ids = input_ids

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # expand encoder_outputs
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs
            )

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["encoder_input_ids"] = encoder_input_ids

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        output = self._generate_no_beam_search(
            input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            attention_mask=attention_mask,
            use_cache=use_cache,
            id_pairs_down=id_pairs_down,
            id_pairs_up=id_pairs_up,
            model_kwargs=model_kwargs,
        )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        id_pairs_down,
        id_pairs_up,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = self(**model_inputs, return_dict=True)
            # calling forward here
            # import ipdb; ipdb.set_trace()
            #outputs.logits (batch, seq_len, input_seq_len)
            next_token_logits = outputs.logits[:, -1, :]

            cur_tok_id = input_ids[0, -1].item()
            down_words_ids = []
            if id_pairs_down:
                if cur_tok_id in id_pairs_down:
                    down_words_ids.extend(id_pairs_down[cur_tok_id])
            up_words_ids = []
            up_words_ids2 = []

            if cur_tok_id == 2385:  # Jr.
                up_words_ids2.append(4)
            # # import ipdb; ipdb.set_trace()
            if id_pairs_up:
                if cur_tok_id in id_pairs_up:
                    up_words_ids.extend(id_pairs_up[cur_tok_id])
                # import ipdb; ipdb.set_trace()

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                down_words_ids=down_words_ids,
                up_words_ids=up_words_ids,
                up_words_ids2=up_words_ids2,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + \
                    (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat(
                [input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
                    eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(
                    is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

    # Function from `generation_utils.py` of Transformers library
    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        down_words_ids,
        up_words_ids,
        up_words_ids2,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        # other case, only replace <arg>
        ori_next_token = torch.argmax(scores, dim=-1)
        # down_words_ids
        for tok_id in down_words_ids:
            if ori_next_token == tok_id:
                scores[:, tok_id] /= 100
                # scores[:, 50265] = scores[:, 50265].abs()*100
        # up_words_ids
        for tok_id in up_words_ids2:
            scores[:, tok_id] = scores[:, tok_id].abs()*100
        if ori_next_token != 50265:
            for tok_id in up_words_ids:
                scores[:, tok_id] = scores[:, tok_id].abs()*1000

        return scores
