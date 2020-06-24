from typing import Optional
import numpy as np
from overrides import overrides
import torch
from allennlp.training.metrics import CategoricalAccuracy,F1Measure
import sacrebleu


from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

@Metric.register("BELU")
class BELU(Metric):
    def __init__(self) -> None:
        super().__init__()
        self._total_sum = 0
        self.belu_score = 0

        self.pred = []
        self.gold = []

    @overrides
    def __call__(self,pred,target):
        # refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
        #         ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
        # sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

        for p,t in zip(pred,target):
            # Assumes first and last car are artificial in the target
            t = self.remove_boundries(t)
            p = " ".join(p)
            self.pred.append(p)
            self.gold.append(t)

            self._total_sum += 1

    def remove_boundries(self, t):
        x = t.split()[1:-1]
        x = " ".join(x)
        return x

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self._total_sum == 0:
            raise RuntimeError("You never call this metric before.")
        belu = None
        if reset:
            belu = sacrebleu.corpus_bleu(self.pred, [self.gold])
            belu = belu.score
            self.reset()
        return belu

    @overrides
    def reset(self):
        self._total_sum = 0
        self.pred = []
        self.gold = []


# @Metric.register("store_info")
# class BELU(Metric):
#     def __init__(self) -> None:
#         super().__init__()
#         self.attention = []
#
#     @overrides
#     def __call__(self,att):
#         x = att.detach().cpu().numpy()
#         self.attention.append(x)
#
#     def get_metric(self, reset: bool = False):
#         if len(self.attention) == 0:
#             raise RuntimeError("You never call this metric before.")
#         if reset:
#             json.dumps()
#             belu = belu.score
#             self.reset()
#         return None
#
#     @overrides
#     def reset(self):
#         self._total_sum = 0
#         self.pred = []
#         self.gold = []