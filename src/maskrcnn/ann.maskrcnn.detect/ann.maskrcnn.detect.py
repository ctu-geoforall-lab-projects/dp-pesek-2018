#!/usr/bin/env python3
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
#% description: Detect features in images using a Mask R-CNN model
#% keyword: ann
#% keyword: vector
#% keyword: raster
#%end
#%option G_OPT_M_DIR
#% key: images_directory
#% label: Path to a directory with images to detect
#% required: yes
#%end
#%option
#% key: images_format
#% type: string
#% label: Format suffix of images
#% description: .jpg, .tiff, .png, etc.
#% required: yes
#%end
#%option
#% key: model
#% type: string
#% label: Path to the .h5 file containing the model
#% required: yes
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
#% key: masks_output
#% label: Directory where masks will be saved
#% description: keep empty to use just temporary files
#% required: no
#%end
#%option
#% key: output_type
#% type: string
#% label: Type of output
#% options: area, point
#% answer: area
#% required: no
#%end


import os
from shutil import copyfile
import skimage.io
import sys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import grass.script as gscript
from grass.script.utils import get_lib_path

path = get_lib_path(modname='maskrcnn', libname='model')
if path is None:
    grass.script.fatal('Not able to find the maskrcnn library directory.')
sys.path.append(path)


def main(options, flags):

    import model as modellib
    from config import ModelConfig

    try:
        imagesDir = options['images_directory']
        modelPath = options['model']
        classes = options['classes'].split(',')
        outputType = options['output_type']
        if options['images_format'][0] != '.':
            format = '.{}'.format(options['images_format'])
        else:
            format = options['images_format']
        masksDir = options['masks_output']
        if masksDir == '':
            masksDir = gscript.core.tempfile().rsplit(os.sep, 1)[0]
    except KeyError:
        # GRASS parses keys and values as bytes instead of strings
        imagesDir = options[b'images_directory'].decode('utf-8')
        modelPath = options[b'model'].decode('utf-8')
        classes = options[b'classes'].decode('utf-8').split(',')
        outputType = options[b'output_type'].decode('utf-8')
        if options[b'images_format'].decode('utf-8')[0] != '.':
            format = '.{}'.format(options[b'images_format'].decode('utf-8'))
        else:
            format = options[b'images_format'].decode('utf-8')
        masksDir = options[b'masks_output'].decode('utf-8')
        if masksDir == '':
            masksDir = gscript.core.tempfile().decode('utf-8').rsplit(os.sep,
                                                                      1)[0]

    # TODO: (3 different brands in case of lot of classes?)
    if len(classes) > 255:
        raise SystemExit('Too many classes. Must be less than 256.')

    classesColours = range(len(classes) + 1)

    if len(set(classes)) != len(classes):
        raise SystemExit('ERROR: Two or more classes have the same name.')

    # Create model object in inference mode.
    config = ModelConfig(numClasses=len(classes) + 1)
    model = modellib.MaskRCNN(mode="inference", model_dir=modelPath,
                              config=config)

    model.load_weights(modelPath, by_name=True)

    # TODO: Use the whole list instead of iteration
    for imageFile in [file for file in next(
            os.walk(imagesDir))[2] if os.path.splitext(file)[1] == format]:
        image = skimage.io.imread(os.path.join(imagesDir, imageFile))

        # Run detection
        results = model.detect([image], verbose=1)

        # Save results
        for r in results:
            save_instances(image,
                           r['rois'],
                           r['masks'],
                           r['class_ids'],
                           # ['BG'] + [i for i in classes],
                           # r['scores'],
                           outputDir=masksDir,
                           which=outputType,
                           title=imageFile,
                           colours=classesColours)

    print('Masks detected. Georeferencing masks...')
    masks = list()
    detectedClasses = list()
    for referencing in [file for file in next(
            os.walk(imagesDir))[2] if (
                os.path.splitext(file)[1] != format and format in file)]:
        fileName, refExtension = referencing.split(format)
        # TODO: Join with converting to one loop
        for i in range(1, len(classes) + 1):
            maskName = fileName + '_' + str(i)
            maskFileName = maskName + '.png'
            if os.path.isfile(os.path.join(masksDir, maskFileName)):
                if i not in detectedClasses:
                    detectedClasses.append(i)
                masks.append(maskName)
                copy_georeferencing(imagesDir, masksDir, maskFileName,
                                    refExtension, referencing)

                gscript.run_command('r.in.gdal',
                                    input=os.path.join(masksDir, maskFileName),
                                    output=maskName,
                                    band=1,  # TODO: 3 if 3 band masks
                                    overwrite=gscript.overwrite(),
                                    quiet=True)

    print('Converting masks to vectors...')
    masksString = ','.join(masks)
    for i in detectedClasses:
        print('Processing {} map...'.format(classes[i - 1]))
        for maskName in masks:
            gscript.run_command('g.region',
                                raster=maskName,
                                quiet=True)
            gscript.run_command('r.mask',
                                raster=maskName,
                                maskcats=classesColours[i],
                                quiet=True)
            gscript.run_command('r.to.vect',
                                's',
                                input=maskName,
                                output=maskName,
                                type=outputType,
                                quiet=True)
            gscript.run_command('r.mask',
                                'r',
                                quiet=True)

        gscript.run_command('v.patch',
                            input=masksString,
                            output=classes[i - 1])
        gscript.run_command('g.remove',
                            'f',
                            name=masksString,
                            type='vector',
                            quiet=True)
    gscript.run_command('g.remove',
                        'f',
                        name=masksString,
                        type='raster',
                        quiet=True)


def copy_georeferencing(imagesDir, masksDir, maskFileName, refExtension,
                        referencing):
    r2 = os.path.join(masksDir, maskFileName + refExtension)
    copyfile(os.path.join(imagesDir, referencing), r2)


def apply_mask(image, mask, colour):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  np.zeros([image.shape[0],
                                            image.shape[1]]) + colour[c],
                                  image[:, :, c])
    return image


def save_instances(image,
                   boxes,
                   masks,
                   class_ids,
                   # class_names,
                   # scores=None,
                   title="",
                   # figsize=(16, 16),
                   # ax=None,
                   outputDir='',
                   which='mask',
                   colours=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.

    May be extended in the future (commented parameters)
    """

    dpi = 80
    height, width = image.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to detect in image {}*** \n".format(title))
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    if which == 'area':
        for classId in set(class_ids):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            masked_image = np.zeros(image.shape)
            index = 0

            for i in range(N):
                if class_ids[i] != classId:
                    continue
                colour = (colours[class_ids[i]],) * 3

                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox.
                    # Likely lost in image cropping.
                    continue

                # Mask
                mask = masks[:, :, i]
                masked_image = apply_mask(masked_image, mask, colour)

                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image
                # edges.
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor='none')
                    ax.add_patch(p)

                index = i
                # TODO: write probabilities
                # score = scores[i] if scores is not None else None

            ax.imshow(masked_image.astype(np.uint8), interpolation='nearest')
            ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
            plt.savefig(
                os.path.join(
                    outputDir,
                    os.path.splitext(title)[0] + '_' + str(class_ids[index])),
                dpi=dpi)
            plt.close()
    elif which == 'point':
        for classId in set(class_ids):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            masked_image = np.zeros(image.shape)
            index = 0
            for i in range(N):
                if class_ids[i] != classId:
                    continue

                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')

                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox.
                    # Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                masked_image[int((y1 + y2) / 2)][int((x1 + x2) / 2)] = colours[
                    class_ids[i]]

                index = i

                # TODO: write probabilities
                # Label
                # class_id = class_ids[i]
                # score = scores[i] if scores is not None else None
                # label = class_names[class_id]

            ax.imshow(masked_image.astype(np.uint8), interpolation='nearest')
            ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

            plt.savefig(
                os.path.join(
                    outputDir,
                    os.path.splitext(title)[0] + '_' + str(class_ids[index])),
                dpi=dpi)
            # plt.show()
            plt.close()


if __name__ == "__main__":
    options, flags = gscript.parser()
    main(options, flags)
