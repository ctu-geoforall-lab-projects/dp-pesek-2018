#!/usr/bin/env python
#
############################################################################
#
# MODULE:	    ann.maskrcnn.train
# AUTHOR(S):	Ondrej Pesek <pesej.ondrek@gmail.com>
# PURPOSE:	    Train your Mask R-CNN network
# COPYRIGHT:	(C) 2017 Ondrej Pesek and the GRASS Development Team
#
#		This program is free software under the GNU General
#		Public License (>=v2). Read the file COPYING that
#		comes with GRASS for details.
#
#############################################################################

#%module
#% description: Train your Mask R-CNN network
#% keyword: ann
#% keyword: vector
#% keyword: raster
#%end
#%flag
#%  key: e
#%  description: Pretrained weights were trained on another classes / resolution / sizes
#%end
#%flag
#%  key: s
#%  description: Save also a list of images unused for training to logs dir
#%end
#%option G_OPT_M_DIR
#% key: training_dataset
#% label: Path to the dataset with images and masks
#% required: yes
#%end
#%option
#% key: model
#% type: string
#% label: Path to the .h5 file to use as initial values
#% description: Keep empty to train from a scratch
#% required: no
#% multiple: no
#%end
#%option
#% key: classes
#% type: string
#% label: Names of classes separated with ","
#% required: yes
#% multiple: yes
#%end
#%option G_OPT_M_DIR
#% key: logs
#% label: Path to the directory in which will be models saved
#% required: yes
#%end
#%option
#% key: name
#% type: string
#% label: Name for output models
#% required: yes
#%end
#%option
#% key: epochs
#% type: integer
#% label: Number of epochs
#% required: no
#% multiple: no
#% answer: 200
#% guisection: Training parameters
#%end
#%option
#% key: steps_per_epoch
#% type: integer
#% label: Steps per each epoch
#% required: no
#% multiple: no
#% answer: 3000
#% guisection: Training parameters
#%end
#%option
#% key: rois_per_image
#% type: integer
#% label: How many ROIs train per image
#% required: no
#% multiple: no
#% answer: 64
#% guisection: Training parameters
#%end
#%option
#% key: images_per_gpu
#% type: integer
#% label: Number of images per GPU
#% description: Bigger number means faster training but needs a bigger GPU
#% required: no
#% multiple: no
#% answer: 1
#% guisection: Training parameters
#%end
#%option
#% key: gpu_count
#% type: integer
#% label: Number of GPUs to be used
#% required: no
#% multiple: no
#% answer: 1
#% guisection: Training parameters
#%end


import grass.script as gscript
from grass.pygrass.utils import get_lib_path
import sys
import os
from subprocess import call

path = get_lib_path(modname='maskrcnn', libname='py3train')
if path is None:
    grass.script.fatal('Not able to find the maskrcnn library directory.')


###########################################################
# unfortunately, it needs python3, see file py3train.py
###########################################################
# sys.path.append(path)
# from configs import ModelConfig


def main(options, flags):

    dataset = options['training_dataset']
    initialWeights = options['model']
    classes = options['classes']
    name = options['name']
    logs = options['logs']
    epochs = int(options['epochs'])
    stepsPerEpoch = int(options['steps_per_epoch'])
    ROIsPerImage = int(options['rois_per_image'])
    imagesPerGPU = int(options['images_per_gpu'])
    GPUcount = int(options['gpu_count'])

    flagsString = ''
    for flag, value in flags.items():
        if value is True:
            flagsString += flag

    ###########################################################
    # unfortunately, redirect everything to python3
    ###########################################################
    call('python3 {}{}py3train.py --dataset={} --model={} --logs={} '
         '--name={} --images_per_gpu={} --gpu_count={} --epochs={} '
         '--steps_per_epoch={} --classes={} --rois_per_image={} '
         '--flags={}'.format(
            path, os.sep,
            dataset,
            initialWeights,
            logs,
            name,
            imagesPerGPU,
            GPUcount,
            epochs,
            stepsPerEpoch,
            classes,
            ROIsPerImage,
            flagsString),
         shell=True)


if __name__ == "__main__":
    options, flags = gscript.parser()
    main(options, flags)
