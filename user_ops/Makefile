all:

include $(KALDI_ROOT)/src/kaldi.mk

TF_INC := $(shell python -c "import tensorflow as tf; print(tf.sysconfig.get_include())")

OBJFILES = kaldi.o

LIBNAME = kaldi

CXXFLAGS += -I$(TF_INC) -I$(KALDI_ROOT)/src

ifeq ($(shell uname), Darwin)
    LDFLAGS += -undefined dynamic_lookup
else
    LDFLAGS += -Wl,-warn-unresolved-symbol
endif

LDFLAGS += -L$(KALDI_ROOT)/src/lib
LDFLAGS += -lkaldi-decoder -lkaldi-matrix -lkaldi-base -lkaldi-lat -lkaldi-util
LDFLAGS += -lkaldi-fstext -lkaldi-hmm -lkaldi-feat

include $(KALDI_ROOT)/src/makefiles/default_rules.mk