import os
import sys
import numpy
import matplotlib.pyplot as plt



# /home/yuyang/workspace/RetinaViT/workdirs/rv_attn_mag.o128921573/11-19_0919
# /home/yuyang/workspace/RetinaViT/workdirs/rv_attn_more.o128921646/11-19_1047

attention_file_directory = sys.argv[1]

for key in ["attn_distribution", "attn", "before_mlp", "query", "key", "value"]:
    file_name = f"{key}.npy"
    value = numpy.load(os.path.join(attention_file_directory, file_name))

    average = numpy.average(value, axis=0)

    plt.clf()
    plt.plot(average)
    plt.savefig(f"{key}.png")
