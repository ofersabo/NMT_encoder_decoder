from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.attention import BilinearAttention
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import RegularizerApplicator, util
from overrides import overrides
from my_library.my_loss_metric import BELU
import os, json

@Model.register("attention")
class SequenceToSequence(Model):
    """
    Base class for sequence-to-sequence models.
    """
    DECODERS = {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}
    def __init__(self,
                 # Vocabluary.
                 vocab: Vocabulary,

                 # Embeddings.
                 source_text_field_embedder: TextFieldEmbedder,
                 target_embedding_size: int,

                 hidden_size: int,
                 decoder_type: str = "gru",
                 source_namespace: str = "tokens",
                 target_namespace: str = "target",

                 # Hyperparamters and flags.
                 drop_out_rate: float = 0.0,

                 decoder_attention_function: BilinearAttention = None,
                 decoder_is_bidirectional: bool = False,
                 decoder_num_layers: int = 1,
                 apply_attention: bool = False,
                 max_decoding_steps: int = 100,
                 # scheduled_sampling_ratio: float = 0.0,
                 attention_file: str = "attention_data.jsonl",

                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        assert decoder_type in SequenceToSequence.DECODERS

        self.source_vocab_size = vocab.get_vocab_size(source_namespace)
        self.target_vocab_size = vocab.get_vocab_size(target_namespace)
        self.source_field_embedder = source_text_field_embedder
        self.encoder = torch.nn.LSTM(self.source_field_embedder.get_output_dim(), hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.metrics = {"BELU": BELU()}

        self.target_vocab_size = vocab.get_vocab_size(target_namespace)
        self.target_embedder = Embedding(self.target_vocab_size, target_embedding_size)

        if apply_attention:
            decoder_input_size = target_embedding_size + hidden_size
        else:
            decoder_input_size = target_embedding_size + hidden_size

        # self.analyze_this_target = START_SYMBOL + " S T A I R C A S E . . . " + END_SYMBOL
        self.attention_file = attention_file

        self.dropout = torch.nn.Dropout(p=drop_out_rate)
        # Hidden size of the encoder and decoder should match.
        decoder_hidden_size = hidden_size
        self.decoder = SequenceToSequence.DECODERS[decoder_type](
            decoder_input_size,
            decoder_hidden_size,
            num_layers=decoder_num_layers,
            batch_first=True,
            bias=True,
            bidirectional=decoder_is_bidirectional
        )
        self.output_projection_layer = torch.nn.Linear(hidden_size, len(vocab._token_to_index['target']))
        self.apply_attention = apply_attention
        self.decoder_attention_function = decoder_attention_function or BilinearAttention(
            matrix_dim=hidden_size,
            vector_dim=hidden_size
        )

        # Hyperparameters.
        self._max_decoding_steps = max_decoding_steps
        # self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._decoder_is_lstm = isinstance(self.decoder, torch.nn.LSTM)
        self._decoder_is_gru = isinstance(self.decoder, torch.nn.GRU)
        self._decoder_num_layers = decoder_num_layers

        self._start_index = vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = vocab.get_token_index(END_SYMBOL, target_namespace)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self.count = 0
        self.first_dump = True
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    @overrides
    def forward(self,source, source_clean, target = None ,target_clean = None,analyze_instance = False) -> Dict[str, torch.Tensor]:
        # couldn't run this with batch larger than 1
        # if self._decoder_is_gru:
        # self.decoder.flatten_parameters()
        attentions_to_keep = []
        source_sequence_encoded = self.encode_input(source)

        source_encoded = source_sequence_encoded[:, -1]

        batch_size = source_encoded.size(0)

        # Determine number of decoding steps. If training or computing validation, we decode
        # target_seq_len times and compute loss.
        if target:
            target_tokens = target['tokens']
            target_seq_len = target['tokens'].size(1)
            num_decoding_steps = target_seq_len - 1
        else:
            num_decoding_steps = self.max_decoding_steps

        # last_predictions = None
        step_logits, step_probabilities, step_predictions = [], [], []
        decoder_hidden = self.init_decoder_hidden_state(source_encoded)
        for timestep in range(num_decoding_steps):
            if self.training:
                input_choices = target_tokens[:, timestep]
            else:
                if timestep == 0:  # Initialize decoding with the start token.
                    input_choices = (torch.ones((batch_size,)) * self._start_index).long()
                else:
                    input_choices = last_predictions
            decoder_input, the_attention = self.prepare_decode_step_input(input_choices, decoder_hidden,
                                                                          source_sequence_encoded, source_encoded)
            # if len(decoder_input.shape) < 3:
            decoder_input = decoder_input.unsqueeze(1)

            _, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Probability distribution for what the next decoded class should be.
            output_projection = self.output_projection_layer(decoder_hidden[0][-1]
                                                             if self._decoder_is_lstm
                                                             else decoder_hidden[-1])
            step_logits.append(output_projection.unsqueeze(1))

            # Collect predicted classes and their probabilities.
            # class_probabilities = F.softmax(output_projection, dim=-1)
            _, predicted_classes = torch.max(output_projection, 1)
            # step_probabilities.append(class_probabilities.unsqueeze(1))
            step_predictions.append(predicted_classes.unsqueeze(1))
            last_predictions = predicted_classes
            if analyze_instance and self.apply_attention:
                attentions_to_keep.append(the_attention[0].detach().cpu().tolist())

        logits = torch.cat(step_logits, 1)
        # class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"logits": logits,
                       # "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if target:
            target_mask = util.get_text_field_mask(target)
            relevant_targets = target['tokens'][:, 1:]
            relevant_mask = target_mask[:, 1:]
            loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
            output_dict["loss"] = loss
            self.decode(output_dict)
            self.metrics['BELU'](output_dict["predicted_tokens"], target_clean)
            if analyze_instance:
                line = {"source": source_clean, "target": target_clean, "attention": attentions_to_keep}
                self.write_to_file(line)

        return output_dict

    def encode_input(self, source: Dict[str, torch.LongTensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Required shapes: (batch_size, sequence_length, decoder_hidden_size)
        """
        source_sequence_embedded = self.source_field_embedder(source).to(self.device)
        source_sequence_embedded = self.dropout(source_sequence_embedded)

        encoded_source_sequence = self.encoder(source_sequence_embedded)
        return encoded_source_sequence[0]

    def init_decoder_hidden_state(self, source_sequence_encoded: torch.FloatTensor) -> torch.FloatTensor:
        """
        Required shape: (batch_size, num_decoder_layers, encoder_hidden_size)
        """
        decoder_primer = source_sequence_encoded.unsqueeze(0)
        decoder_primer = decoder_primer.expand(
            self._decoder_num_layers, -1, self.encoder.hidden_size
        )

        # If the decoder is an LSTM, we need to initialize a cell state.
        if self._decoder_is_lstm:
            decoder_primer = (decoder_primer, torch.zeros_like(decoder_primer))

        return decoder_primer

    def prepare_decode_step_input(self,
                                  input_indices: torch.LongTensor,
                                  decoder_hidden: torch.LongTensor,
                                  encoder_outputs: torch.LongTensor,
                                  source_encoded: torch.LongTensor
                                  ) -> torch.LongTensor:
        """
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden : torch.LongTensor, optional (not needed if no attention)
            Output from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        """
        embedded_input = self.target_embedder(input_indices.to(self.device))
        if self.apply_attention:
            if isinstance(decoder_hidden, tuple):
                decoder_hidden = decoder_hidden[0]
            input_weights = self.decoder_attention_function(decoder_hidden[-1], encoder_outputs)

            # (batch_size, encoder_output_dim)
            attended_input = util.weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1), input_weights
        else:
            return torch.cat((source_encoded, embedded_input), -1), None

    # @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first END_SYMBOL.
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return all_predicted_tokens

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.get_metric(reset)
            if x is not None:
                metric_results[metric_name] = x

        return metric_results

    def write_to_file(self, json_instance):
        if not self.first_dump:
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
            self.first_dump = False
        with open(self.attention_file, append_write) as fw:
            # for m,values in data_results.items():
            fw.write(json.dumps(json_instance) + "\n")
            # fw.write("\n")
