import sys,json
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

x_axis = None
y_axis = None
rows = []
file_to_open = sys.argv[1] if len(sys.argv) > 1 else "attention_data.jsonl"
with open(file_to_open) as fr:
    lines = fr.readlines()
for epoch_index,l in enumerate(lines,1):
    data = json.loads(l.strip())
    x = data["source"]
    y = data["target"]
    if x_axis:
        assert x == x_axis
        assert y == y_axis
    x_axis = x
    y_axis = y
    row = np.array(data['attention'])
    x = x[0]
    y = y[0]
    sub_title = " ".join(x.split()[1:-1]) +" ->\n"+ " ".join(y.split()[1:-1])
    plt.figure(epoch_index)
    full_title = "attention based alignment:\n" + sub_title
    # ax.set_title(full_title)
    plt.title(full_title,fontsize=8)
    ax = sns.heatmap(row, xticklabels=x.split(), yticklabels=y.split()[1:],cmap="Blues")
    # ax = sns.heatmap(row, linewidth=0.5,yticklabels=y)
    # ax.set_yticks(y[0].split()[1:])
    plt.savefig("heat_map_epoch_"+str(epoch_index)+".png")