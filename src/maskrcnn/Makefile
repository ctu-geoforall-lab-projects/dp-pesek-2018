MODULE_TOPDIR =../../..

PGM = maskrcnn

SUBDIRS = maskrcnnlib \
	ann.maskrcnn.train \
	ann.maskrcnn.detect \

include $(MODULE_TOPDIR)/include/Make/Dir.make

default: parsubdirs htmldir

install: installsubdirs
	$(INSTALL_DATA) $(PGM).html $(INST_DIR)/docs/html/
