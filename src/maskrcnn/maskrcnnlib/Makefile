MODULE_TOPDIR = ../../..

include $(MODULE_TOPDIR)/include/Make/Other.make
include $(MODULE_TOPDIR)/include/Make/Python.make

MODULES = __init__ py3train py3detect \
 config utils model parallel_model visualize

ETCDIR = $(ETC)/maskrcnn

PYFILES := $(patsubst %,$(ETCDIR)/%.py,$(MODULES))
PYCFILES := $(patsubst %,$(ETCDIR)/%.pyc,$(MODULES))

default: $(PYFILES) $(PYCFILES)

$(ETCDIR):
	$(MKDIR) $@

$(ETCDIR)/%: % | $(ETCDIR)
	$(INSTALL_DATA) $< $@

install:
	cp -r $(ETCDIR) $(INST_DIR)/etc
