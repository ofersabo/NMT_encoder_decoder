from typing import Dict, List
import collections
import logging
import math
import allennlp
import torch
from overrides import overrides
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from torch.autograd import Variable
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from torch.nn.parameter import Parameter
from allennlp.nn import util
from torch.nn.functional import softmax, normalize
from my_library.my_loss_metric import BELU
from allennlp.modules.token_embedders import embedding

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('RNN')
class Rnn(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_text_field_embedder: TextFieldEmbedder,
                 target_text_field_embedder: TextFieldEmbedder,
                 encoder,
                 hidden_size: int,
                 metrics: Dict[str, allennlp.training.metrics.Metric] = None,
                 regularizer: RegularizerApplicator = None,
                 bidirectional: bool = False,

                 ) -> None:
        super().__init__(vocab, regularizer)

        self.source_embeddings = source_text_field_embedder
        self.target_embeddings = target_text_field_embedder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or {
            "BELU": BELU(),
        }
        self.tanh = torch.nn.Tanh()
        is_bidirectional = bidirectional
        self.enconder = torch.nn.LSTM(self.source_embeddings.get_output_dim(), hidden_size, num_layers=2, bidirectional=is_bidirectional, batch_first=True)
        self.decoder_cell = torch.nn.LSTMCell(hidden_size * (1 + is_bidirectional) + self.source_embeddings.get_output_dim(), hidden_size)
        self.final_layer = torch.nn.Linear(hidden_size,len(vocab._token_to_index['target']))
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.maximal_possible_output = 100
        self.count = 0
        self.char_mapping = self.vocab._index_to_token['target']
        self.token2index = self.vocab.get_token_to_index_vocabulary()
        self.start_token = self.token2index['<s>']
        self.end_token = self.token2index['<\\s>']

    @overrides
    def forward(self,source, source_clean, target = None ,target_clean = None) -> Dict[str, torch.Tensor]:
        output_dict = {}
        batch_size = source['tokens'].size(0)
        # Encoder
        encoder_output = self.get_encoder_last_cell(source)

        # Decoder
        char_predictions = torch.ones(size=(1,batch_size)).long().to(self.device) * self.start_token
        if target:
            input_embedding = self.target_embeddings(target)
            index_to_predict = target['tokens'][:,1:] # shift the target so we have the next char at each point

        # I decided that the start char embedding is vectors of zeros
        current_char_embedding = torch.zeros((batch_size,self.source_embeddings.get_output_dim())).to(self.device)

        loss = torch.tensor([0.]).to(self.device)
        next_cell_state = None
        which_instance_ended = torch.tensor([0] * batch_size).to(dtype=bool,device=self.device)
        for i in range(self.maximal_possible_output):
            input_to_cell_word_with_encoder = self.concat_two_tensors(encoder_output,current_char_embedding)
            hx, cx = self.decoder_cell(input_to_cell_word_with_encoder, next_cell_state)
            scores_for_each_char = self.final_layer(hx)
            if target:
                label_char_to_predict = index_to_predict[:,i] if i < index_to_predict.size(1) else \
                    torch.zeros_like(index_to_predict[:,0])
                nll = self.crossEntropyLoss(scores_for_each_char,label_char_to_predict)
                loss += nll

            which_instance_ended = self.did_seq_end(label_char_to_predict, scores_for_each_char, which_instance_ended)


            #remove unnecessary examples
            encoder_output, index_to_predict, input_embedding, label_char_to_predict, next_cell_state = self.removed_ended_seq(
                cx, encoder_output, hx, index_to_predict, input_embedding, label_char_to_predict,target, which_instance_ended)

            predicted_char = scores_for_each_char.argmax(dim=1).detach().long()
            char_predictions = torch.cat([char_predictions, predicted_char.unsqueeze(0)],dim=0)
            current_char_embedding = self.get_embedding_of_next_char(current_char_embedding, i, index_to_predict,input_embedding, predicted_char)

            if which_instance_ended.sum().item() == batch_size:
                break

        # char_predictions = char_predictions[1:]
        str_of_pred = self.get_string_from_predction(batch_size, char_predictions)

        self.metrics['BELU'](str_of_pred, target_clean)
        output_dict["loss"] = loss
        output_dict["predict"] = char_predictions

        if self.count % 200 == 0 or not self.training:
            print("pred ", str_of_pred,flush=True)
            print("gold ", target_clean,flush=True)

        self.count += 1

        return output_dict

    def get_string_from_predction(self, batch_size, char_predictions):
        the_pred_strings = self.decode_prediction(char_predictions, batch_size)
        str_of_pred = self.convert_list_to_string(the_pred_strings)
        return str_of_pred

    def get_embedding_of_next_char(self, current_char_embedding, i, index_to_predict, input_embedding, predicted_char):
        if self.training:
            # force teaching
            current_char_embedding = input_embedding[:, i + 1] if i < index_to_predict.size(1) else input_embedding[:,
                                                                                                    -1]
        else:
            # get the output of previous char_predictions
            current_char_embedding = self.target_embeddings({"tokens": predicted_char}).to(self.device)
        return current_char_embedding

    def removed_ended_seq(self, cx, encoder_output, hx, index_to_predict, input_embedding, label_char_to_predict,
                          target, which_instance_ended):
        encoder_output = encoder_output[~which_instance_ended]
        index_to_predict = index_to_predict[~which_instance_ended]
        next_cell_state = (hx[~which_instance_ended], cx[~which_instance_ended])
        if target:
            input_embedding = input_embedding[~which_instance_ended]
        label_char_to_predict = label_char_to_predict[~which_instance_ended]
        return encoder_output, index_to_predict, input_embedding, label_char_to_predict, next_cell_state

    def did_seq_end(self, label_char_to_predict, scores_for_each_char, which_instance_ended):
        if self.training:
            which_instance_ended = which_instance_ended + (label_char_to_predict == 3)
        else:
            which_instance_ended = which_instance_ended + (scores_for_each_char.argmax(dim=1) == 3)
        return which_instance_ended

    def get_encoder_last_cell(self, source):
        encoder_embedding = self.source_embeddings(source)
        lstm_output = self.enconder(encoder_embedding)[0]
        encoder_output = lstm_output[:, -1]
        return encoder_output

    def concat_two_tensors(self, first, second):
        catted = torch.cat((first, second), dim=-1)
        return catted

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_results = {}
        for metric_name, metric in self.metrics.items():
            metric_results[metric_name] = metric.get_metric(reset)

        return metric_results


    def decode_prediction(self, pred,batch_size):
        strings = []
        for _ in range(batch_size):
            strings.append([])
        for location_in_word in pred.tolist():
            for s_id,c in enumerate(location_in_word):
                this_word = strings[s_id]
                if len(this_word) > 0 and this_word[-1] == -1:
                    continue
                the_char = self.char_mapping[c]
                strings[s_id].append(the_char)
                if c == 3:
                    strings[s_id].append(-1)

        for l in range(len(strings)):
            if strings[l][-1] == -1:
                strings[l] = strings[l][:-1]

        return strings

    def convert_list_to_string(self, strings):
        l = []
        for _ in range(len(strings)):
            l.append([])
        for s in range(len(strings)):
            l[s] = " ".join(strings[s])
        return l

    # @overrides
    # def forward(self,  sentences, locations):
    #     bert_context_for_relation = self.source_embeddings(sentences)
    #     bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, locations)
    #
    #     after_mlp_aggregated = self.go_thorugh_mlp(bert_represntation,self.first_liner_layer,self.second_liner_layer).to(self.device)
    #     try:
    #         x = self.no_relation_vector
    #         try:
    #             x = self.nota_value
    #             return {"vector": after_mlp_aggregated}
    #         except:
    #             NOTA = self.go_thorugh_mlp(self.no_relation_vector,self.first_liner_layer,self.second_liner_layer).to(self.device)
    #             return {"vector":after_mlp_aggregated,"NOTA":NOTA}
    #     except:
    #         pass
    #
    #     return {"vector":after_mlp_aggregated}
