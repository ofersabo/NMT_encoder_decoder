import json
import shutil
import sys

from allennlp.commands import main

model_location = sys.argv[1] if len(sys.argv) > 1 else 'best_part_1_reproduce'
src_location = sys.argv[2] if len(sys.argv) > 2 else 'data/test.src'
trg_location = sys.argv[3] if len(sys.argv) > 3 else 'data/test.trg'
output = sys.argv[4] if len(sys.argv) > 4 else "here"

locations = src_location + " " + trg_location

# Use overrides to train on CPU.
# overrides = json.dumps({"trainer":{"cuda_device": 0},"iterator": {"type": "basic", "batch_size": 1},"validation_data_path": locations})
# '{"model":{"cuda_device":[0]},"trainer":{"cuda_device": [0]} }'
overrides = json.dumps({"trainer":{"cuda_device": [-1]},"model":{"cuda_device":[-1]}})
#on_the_fly
# print(model_location)

if output != "":
    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "evaluate",
        model_location,
        locations,
        "--include-package", "my_library",
        "--output-file", output,
        "-o", overrides,
    ]
else:
    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "evaluate",
        model_location,
        locations,
        "--include-package", "my_library",
        # "--cuda-device", 0,
        "-o", overrides,
    ]


main()