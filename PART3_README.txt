# This readme first part is the same as the part2 readme beacuse the attention data is genereated by default.
# I again note that for the attention data to be generated, the dev dataset must contain the "dev" string in the target data.
# The example which we tracked is a random example of the dev set.

# To train the model in part 2 and dump the attention data:
allennlp train experiments/attention.jsonnet -s save_model_to_this_output_dir --include-package my_library

I assume that the trg file of the dev dataset contains "dev" in its file name. that is how the script generates the attention heatmap data.
The raw data of the attention file is dumped into "attention_data.jsonl"

to generate 10 heatmaps of the attention use the script:
python plot_heatmap.py

This generates 10 files named:
heat_map_epoch_*.png

The * means the epoch index, count starts from 1.
