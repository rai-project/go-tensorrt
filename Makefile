CXX := nvcc # This is the main compiler

LDFLAGS := -lnvinfer -lnvcaffe_parser -lQtCore -lQtGui -Lcuda -lcudaUtil
SRCDIR := cbits
BUILDDIR := build
CXXFLAGS := -std=c++11 -I${SRCDIR} -I${SRCDIR}/util -I${SRCDIR}/util/cuda -O3 -I/usr/include/qt4/QtGui -I/usr/include/qt4/QtCore -I/usr/include/qt4 -Icuda
TARGETA := test2
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
INC :=-lnvinfer -lnvcaffe_parser -lQtGui -lQtCore
# TEST := test
hi:
# test:
# 	nvcc  -std=c++11 -Icbits -Icbits/util -Icbits/util/cuda -O3 -I/usr/lib/x86_64-linux-gnu -I/usr/include/qt4/QtGui -I/usr/include/qt4  -lnvinfer -lnvcaffe_parser -lQtGui test.cpp -o build/test.o
# 	nvcc -std=c++11 -Icbits -Icbits/util -Icbits/util/cuda -O3 -I/usr/include/qt4/QtGui -I/usr/include/qt4 -lQtGui -lnvinfer -lnvcaffe_parser -c -o build/util/loadImage.o cbits/util/loadImage.cpp
buildtest: $(OBJECTS) $(OBJECTS2) $(BUILDDIR)/test.o
		@echo " Linking..."
		@echo $(SOURCES)
		@echo $(OBJECTS)
		# @echo $(OBJECTS2)
		@echo " $(CCX) $^ -o $(TARGET) $(LDFLAGS)"; $(CXX) $^ -o $(TARGETA) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
		@mkdir -p $(BUILDDIR)
		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
$(BUILDDIR)/util/%.o: $(SRCDIR)/util/%.$(SRCEXT)
		@mkdir -p $(BUILDDIR)/util
		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
# $(BUILDDIR)/util/%.o: $(SRCDIR)/util/camera/%.$(SRCEXT)
# 		@mkdir -p $(BUILDDIR)/util
# 		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
# $(BUILDDIR)/util/cuda%.o: $(SRCDIR)/util/cuda/%.$(SRCEXT)
# 		@mkdir -p $(BUILDDIR)/util/cuda
# 		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
# $(BUILDDIR)/util/cuda/%.o: $(SRCDIR)/util/cuda/%.cu
# 		@mkdir -p $(BUILDDIR)/util/cuda
# 		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
# $(BUILDDIR)/%.o: $(SRCDIR)/%.cu
# 		@mkdir -p $(BUILDDIR)/
# 		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
# $(BUILDDIR)/util/%.o: $(SRCDIR)/util/display/%.$(SRCEXT)
# 		@mkdir -p $(BUILDDIR)/util
# 		@echo " $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/test.o: test.cpp
		@mkdir -p $(BUILDDIR)
		@echo $(OBJECTS)
		$(CXX) $(CXXFLAGS) $(LDFLAGS) test.cpp -c -o $(BUILDDIR)/test.o


clean:
		@echo " Cleaning..."; 
		@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)
.PHONY: clean
