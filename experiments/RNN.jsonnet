local cuda = [-1];
//local cuda = [0,1,2,3,4,5,6,7];
//local cuda = [0,1,2,3];
//local bert_type = 'bert-base-cased';
local train_data = "data/train.src data/train.trg";
local dev_data = "data/dev.src data/dev.trg";
local test_data = "data/test.src data/test.trg";


local batch_size = 1;
local lr_with_find = 0.0007125761361357518;
local hidden_size = 150;
local embedding_size = 850;
//local instances_per_epoch = null;

{
  "dataset_reader": {
    "type": "the_reader",
    "lazy": false,
    "tokenizer": {
      "type": "word",
      "word_splitter":"just_spaces",
    },
    "target_token_indexers": {
        "type": "single_id",
    },
    "source_token_indexers": {
        "type": "single_id"
    }
  },
  "train_data_path": train_data,
  "validation_data_path": dev_data,
//  "test_data_path": test_data[setup],
 "model": {
        "type": "my_model",
        "decoder_type":"gru",
        "cuda_device":cuda,
        "source_text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_size,
                    "trainable": true
                },
            }
        },
//    "regularizer": [[".*", {"type": "l2", "alpha": 1e-5}]],
//        "target_text_field_embedder": {
//            "token_embedders": {
//                "tokens": {
//                    "type": "embedding",
//                    "embedding_dim": embedding_size,
//                    "trainable": true
//                },
//            }
//        },
        "hidden_size":hidden_size,
        "target_embedding_size":embedding_size,
        "apply_attention":false,
        "drop_out_rate":0.19828927415429567,
    },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": null
  },
    "validation_iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": null
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": lr_with_find
    },
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+BELU",
    "num_epochs": 10,
    "grad_clipping": 1.0,
//    "grad_norm": 2.0,
    "patience": 50,
    "cuda_device": cuda
  }
}