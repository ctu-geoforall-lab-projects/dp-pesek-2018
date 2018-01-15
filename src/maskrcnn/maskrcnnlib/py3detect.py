#!/usr/bin/env python
#
############################################################################
#
# MODULE:	    py3train
# AUTHOR(S):	Ondrej Pesek <pesej.ondrek@gmail.com>
# PURPOSE:	    A python3 script called to train your Mask R-CNN network
# COPYRIGHT:	(C) 2017 Ondrej Pesek and the GRASS Development Team
#
#		This program is free software under the GNU General
#		Public License (>=v2). Read the file COPYING that
#		comes with GRASS for details.
#
#############################################################################


import os
import sys
import random
import math
import argparse
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import utils
import model as modellib
import visualize
from config import ModelConfig


# Create Model and Load Trained Weights

def detect(imagesDir, modelPath, classes, name, masksDir, outputType,
           classesColours, format):
    # Create model object in inference mode.
    config = ModelConfig(name=name, numClasses=len(classes) + 1)
    model = modellib.MaskRCNN(mode="inference", model_dir=modelPath,
                              config=config)

    # Load weights trained on MS-COCO

    model.load_weights(modelPath, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    classNames = ['BG']
    for i in classes:
        classNames.append(i)

    classesColours = [float(i) for i in classesColours.split(',')]

    # ## Run Object Detection

    # TODO: Use the whole list instead of iteration
    for imageFile in [file for file in next(
            os.walk(imagesDir))[2] if os.path.splitext(file)[1] == format]:
        # fileNames = next(os.walk(imagesDir))[2]
        # a = random.choice(fileNames)
        image = skimage.io.imread(os.path.join(imagesDir, imageFile))

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        # TODO: More images -> more indices than [0]
        r = results[0]
        # print('Stat:', image, r['rois'], r['masks'], r['class_ids'],
        #                             classNames, r['scores'])
        # print(a)
        # print('NEXT VISUALIZE')
        visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'],
                                 classNames, r['scores'], outputDir=masksDir,
                                 which=outputType, title=imageFile,
                                 colours=classesColours)

    # sys.stdout.write('ahoj')

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--images_dir', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--classes', required=True,
                        help="Names of classes")
    parser.add_argument('--name', required=True,
                        help='Name of output models')
    parser.add_argument('--masks_dir', required=True,
                        help='Name of output models')
    parser.add_argument('--output_type', required=True,
                        help='Type of output')
    parser.add_argument('--colours', required=True,
                        help='Type of output')
    parser.add_argument('--format', required=True,
                        help='Format of images')

    args = parser.parse_args()

    detect(args.images_dir, args.model, args.classes.split(','), args.name,
           args.masks_dir, args.output_type, args.colours, args.format)
