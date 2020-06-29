Before running the underneath scripts please install the required packages.
pip install -r requirements.txt

# To train the model in part 1 please use this command line:
On CPU:
allennlp train experiments/RNN.jsonnet -s save_model_to_this_output_dir --include-package my_library -f

On GPU device 0:
allennlp train experiments/RNN.jsonnet -s best_part_1_reproduce --include-package my_library -f -o '{"model":{"cuda_device":[0]},"trainer":{"cuda_device":[0]}}'

# To evaluate the model on the test set:
On CPU:
python evaluate.py save_model_to_this_output_dir data/test.src data/test.trg output_text_file_of_results
output_text_file_of_results is a dictionary with BELU and loss values.


The result I got on the DEV set was 62 BELU score.
Worth mentioning that I used a GPU to train the model.
When I tried to reproduce the result on CPU, the result degrades by 5 points. from 62 to 57.
