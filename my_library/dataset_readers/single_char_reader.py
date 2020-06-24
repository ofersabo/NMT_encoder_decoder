from typing import Dict
from typing import List
import json
import logging
import os
import random
import copy
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ListField, IndexField, MetadataField, Field


from allennlp.common.util import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

start_symbol = START_SYMBOL + " "
end_symbol = " " + END_SYMBOL

@DatasetReader.register("the_reader")
class MTBDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 source_token_indexers = None,
                 target_token_indexers = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self.source_token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace="target")}
        # self._tokenizer.add_special_tokens({'additional_special_tokens':[head_start_token,head_end_token,tail_start_token,tail_end_token])

    @overrides
    def _read(self, file_path):
        files = file_path.split()
        if len(files) == 2:
            source_file, target_file = files
            logger.info("Reading instances from txt files at: %s and %s", source_file,target_file)
        else:
            source_file, target_file = files[0] ,None
            logger.info("Reading instances from only a signle source file at: %s", source_file)
        with open(cached_path(source_file), "r") as data_file:
            source_data = data_file.readlines()
        if target_file:
            choose_instance_to_analyze = random.randint(0,len(source_data)-1) if "dev" in target_file else -1
            with open(cached_path(target_file), "r") as data_file:
                target_data = data_file.readlines()
            assert len(source_data) == len(target_data)
            for ii,(s,t) in enumerate(zip(source_data,target_data)):
                yield self.text_to_instance(s.strip(),t.strip(),ii == choose_instance_to_analyze)
        else:
            for s in source_data:
                yield self.text_to_instance(s.strip())

    @overrides
    def text_to_instance(self, source: str, target = None,analyze_this_instance : bool = False) -> Instance:  # type: ignore
        source = start_symbol + source + end_symbol
        source_tokens = self._tokenizer.tokenize(source)
        source_field = TextField(source_tokens, self.source_token_indexers)
        source_clean_text_for_debug = MetadataField(source)

        fields = {'source': source_field, 'source_clean':source_clean_text_for_debug}
        '''
        target_part
        '''
        if target:
            target = start_symbol + target + end_symbol
            target_tokens = self._tokenizer.tokenize(target)
            target_field = TextField(target_tokens, self.target_token_indexers)
            target_clean_text_for_debug = MetadataField(target)

            fields['target'] = target_field
            fields['target_clean'] = target_clean_text_for_debug
            if analyze_this_instance:
                fields['analyze_instance'] = MetadataField(True)
        return Instance(fields)
