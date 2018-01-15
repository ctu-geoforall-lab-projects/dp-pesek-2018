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
#% label: Path to the .h5 file to use as initial values
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
#%option
#% key: name
#% type: string
#% label: Name for output models
#% required: yes
#%end
#%option G_OPT_M_DIR
#% key: masks_output
#% label: Directory where masks will be saved
#% required: yes
#%end
#%option
#% key: output_type
#% type: string
#% label: Type of output
#% options: areas, points
#% answer: areas
#% required: no
#%end


import grass.script as gscript
from grass.pygrass.utils import get_lib_path
import sys
import os
from subprocess import call, Popen, check_output
from shutil import copyfile
from random import uniform

path = get_lib_path(modname='maskrcnn', libname='py3detect')
if path is None:
    grass.script.fatal('Not able to find the maskrcnn library directory.')

###########################################################
# unfortunately, it needs python3, see file py3train.py
###########################################################
# sys.path.append(path)
# from configs import ModelConfig


def main(options, flags):

    imagesDir = options['images_directory']
    modelPath = options['model']
    classes = options['classes']
    name = options['name']
    # TODO: Use GRASS temp files
    masksDir = options['masks_output']
    outputType = options['output_type']
    format = options['images_format']

    # TODO: Check if unique
    # TODO: (3 different brands in case of lot of classes?)
    classesColours = [0] + [uniform(0, 1) for _ in range(
        len(classes.split(',')))]

    ###########################################################
    # unfortunately, redirect everything to python3
    ###########################################################
    call('python3 {}{}py3detect.py --images_dir={} --model={} --classes={} '
         '--name={} --masks_dir={} --output_type={} --colours={} '
         '--format={}'.format(
            path, os.sep,
            imagesDir,
            modelPath,
            classes,
            name,
            masksDir,
            outputType,
            ','.join([str(col) for col in classesColours]),
            format),
         shell=True)

    print('Masks detected. Georeferencing masks...')
    masks = list()
    for referencing in [file for file in next(
            os.walk(imagesDir))[2] if os.path.splitext(file)[1] != format]:
        fileName, refExtension = referencing.split(format)
        maskName = fileName + '_mask'
        masks.append(maskName)
        maskFileName = maskName + '.png'
        copy_georeferencing(imagesDir, masksDir, maskFileName, refExtension,
                            referencing)

        gscript.run_command('r.in.gdal',
                            input=os.path.join(masksDir, maskFileName),
                            output=maskName,
                            band=1,  # TODO: Change to 3 if 3 bands masks
                            overwrite=gscript.overwrite())

    print('Converting masks to vectors...')
    masksString = ','.join(masks)
    index = 0
    for i in classesColours[1:]:
        for maskName in masks:
            # TODO: Use r.patch, do following for one colour...
            # TODO: ... in previous loop
            gscript.run_command('g.region',
                                raster=maskName)
            gscript.run_command('r.mask',
                                raster=maskName,
                                maskcats=i * 255)
            gscript.run_command('r.to.vect',
                                input=maskName,
                                output=maskName,
                                type='area')
            gscript.run_command('r.mask',
                                'r')

        gscript.run_command('v.patch',
                            input=masksString,
                            output=classes.split(',')[index - 1])
        gscript.run_command('g.remove',
                            'f',
                            name=masksString,
                            type='vector')

        index += 1


def copy_georeferencing(imagesDir, masksDir, maskFileName, refExtension,
                        referencing):
    r2 = os.path.join(masksDir, maskFileName + refExtension)
    copyfile(os.path.join(imagesDir, referencing), r2)


if __name__ == "__main__":
    options, flags = gscript.parser()
    main(options, flags)
