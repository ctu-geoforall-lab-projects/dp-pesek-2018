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


import grass.script as gscript
from grass.script.utils import get_lib_path
import os
# from subprocess import call
from shutil import copyfile
# from random import randint
import skimage.io
import sys

path = get_lib_path(modname='maskrcnn', libname='model')
if path is None:
    grass.script.fatal('Not able to find the maskrcnn library directory.')
sys.path.append(path)

###########################################################
# unfortunately, it needs python3, see file py3train.py
###########################################################


def main(options, flags):

    # import utils
    import model as modellib
    import visualize
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

    classesWithBG = ['BG'] + [i for i in classes]

    # TODO: Use the whole list instead of iteration
    for imageFile in [file for file in next(
            os.walk(imagesDir))[2] if os.path.splitext(file)[1] == format]:
        image = skimage.io.imread(os.path.join(imagesDir, imageFile))

        # Run detection
        results = model.detect([image], verbose=1)

        # Save results
        for r in results:
            visualize.save_instances(image, r['rois'], r['masks'],
                                     r['class_ids'], classesWithBG,
                                     r['scores'],
                                     outputDir=masksDir, which=outputType,
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
        print('Processing {} map...'.format(classes[i] - 1))
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


if __name__ == "__main__":
    options, flags = gscript.parser()
    main(options, flags)
