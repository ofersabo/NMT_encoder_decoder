import json
import shutil
import sys

from allennlp.commands import main

config_file = "experiments/attention.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer":{"cuda_device": 0},"iterator": {"type": "basic", "batch_size": 1}})
#on_the_fly

serialization_dir = "/tmp/debug_mtb/"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
    "-o", overrides,
]

main()