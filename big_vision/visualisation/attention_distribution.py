import os
import glob
import sys
import numpy
import matplotlib.pyplot as plt

from matplotlib.transforms import blended_transform_factory



JOB_NAME_PREFIX="rv_12_attn."
ROOT_DIR = "/home/yuyang/workspace/RetinaViT/workdirs"
ATTENTION_FILES = ["attn_weight_avg", "attn_mag", "before_mlp"]

for job_id in glob.glob(JOB_NAME_PREFIX + "*", root_dir=ROOT_DIR):
    # /home/yuyang/workspace/RetinaViT/workdirs/rv_12_attn.o146796016/146796016.gadi-pbs
    attention_file_directory = os.path.join(ROOT_DIR, job_id, glob.glob("*gadi-pbs", root_dir=os.path.join(ROOT_DIR, job_id))[0])
    print(f"processing: {attention_file_directory}")

    figure, axes = plt.subplots(12, 3, figsize=(19.2, 12.8))

    for index, key in enumerate(ATTENTION_FILES):
        file_name = f"{key}.npy"
        value = numpy.load(os.path.join(attention_file_directory, file_name))

        # figure.suptitle(key)

        for layer in range(12):
            axis = axes[layer, index]

            layer_values = value[:, layer, :]
            average = numpy.average(layer_values, axis=0)

            axis.plot(average)

            # mark the start of each spatial resolution
            # the lowest two resolution are in one section since they are too close to each other
            x_markers = [0, 4, 20, 84]
            y_markers = [0 for _ in range(4)]
            transform = blended_transform_factory(axis.transData, axis.transAxes)
            axis.scatter(x_markers, y_markers, transform=transform, marker="^", s=16, color="red")

            if index == 0:
                axis.set_ylabel(f"layer_{layer}")

            if layer == 0:
                axes[0, index].set_title(key)

            if layer != 11:
                plt.setp(axis.get_xticklabels(), visible=False) # hide x label in earlier plots

    plt.tight_layout()
    plt.savefig(f"{job_id[len(JOB_NAME_PREFIX):]}.png")
    plt.close()
