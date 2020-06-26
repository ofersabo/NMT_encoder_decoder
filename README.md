# NMT_encoder_decoder exercise

I've used the allennlp framework package. 
This allowed me to easily implement the required code in a very abstractive manner.
 
I attach the requierments.txt file so you could set up your env.
Also if you want to avoid setting up an env you can use my nlp env.

### Part 1  
To run the model in part 1 please use this command line: 
allennlp train experiments/attention.jsonnet -s output_dir --include-package my_library -o '{"model": {"apply_attention":false}}'

### Part 2
allennlp train experiments/attention.jsonnet -s output_dir --include-package my_library
 