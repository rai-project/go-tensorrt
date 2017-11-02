CXX := nvcc # This is the main compiler

# LDFLAGS := -lnvinfer -lnvcaffe_parser -lQtCore -lQtGui
LDFLAGS := -shared
SRCDIR := ./
CXXFLAGS := -std=c++11 -O3 -shared -Xcompiler -fPIC
TARGETA := libcudaUtil.so
SRCEXT := cu
BUILDDIR := ./
SOURCES := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
# INC :=-lnvinfer -lnvcaffe_parser -lQtGui -lQtCore
test:
		@echo $(SOURCES)
		@echo $(OBJECTS)
# 	nvcc  -std=c++11 -Icbits -Icbits/util -Icbits/util/cuda -O3 -I/usr/lib/x86_64-linux-gnu -I/usr/include/qt4/QtGui -I/usr/include/qt4  -lnvinfer -lnvcaffe_parser -lQtGui test.cpp -o build/test.o
# 	nvcc -std=c++11 -Icbits -Icbits/util -Icbits/util/cuda -O3 -I/usr/include/qt4/QtGui -I/usr/include/qt4 -lQtGui -lnvinfer -lnvcaffe_parser -c -o build/util/loadImage.o cbits/util/loadImage.cpp
buildsharedlibrary: $(OBJECTS) 
		@echo " Linking..."
		@echo $(SOURCES)
		@echo $(OBJECTS)
		@echo " $(CCX) $^ -o $(TARGETA) $(LDFLAGS)"; $(CXX) $^ -o $(TARGETA) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
		@mkdir -p $(BUILDDIR)
		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<


clean:
		@echo " Cleaning..."; 
		@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r *.o
		@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r *.so
.PHONY: clean
