# To train the model in part 2 please use this command line:
allennlp train experiments/attention.jsonnet -s save_model_to_this_output_dir --include-package my_library

I assume that the trg file of the dev dataset contains "dev" in its file name. that is how the script generates the attention heatmap data.
The raw data of the attention file is dumped into "attention_data.jsonl"

If you want to generate 10 heatmaps of the attention use the script:
python plot_heatmap.py

# To evaluate the model on the test set:
python evaluate.py save_model_to_this_output_dir data/test.src data/test.trg output_text_file_of_results
output_text_file_of_results is a dictionary with BELU and loss values.


