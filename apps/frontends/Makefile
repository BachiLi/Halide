include ../support/Makefile.inc

CXX_SOURCES = frontends.cpp Lower.cpp
CXX_HEADERS = Lower.h

all: $(BIN)/test

$(BIN)/test: ${CXX_SOURCES} ${CXX_HEADERS}
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) ${CXX_SOURCES} -o $@ -lHalide $(LDFLAGS) $(HALIDE_SYSTEM_LIBS)