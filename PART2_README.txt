# To train the model in part 1 please use this command line:
allennlp train experiments/attention.jsonnet -s save_model_to_this_output_dir --include-package my_library

# To evaluate the model on the test set:
python evaluate.py save_model_to_this_output_dir data/test.src data/test.trg output_text_file_of_results

output_text_file_of_results is a dictionary with BELU and loss values.


