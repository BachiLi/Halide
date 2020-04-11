include ../support/Makefile.inc

CXX_SOURCES = frontends.cpp Lower.cpp StorageFlattening.cpp
CXX_HEADERS = Errors.h StorageFlattening.h Lower.h

all: $(BIN)/test

$(BIN)/test: ${CXX_SOURCES} ${CXX_HEADERS}
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) ${CXX_SOURCES} -o $@ -L$(dir $(LIB_HALIDE)) -lHalide $(LDFLAGS) $(HALIDE_SYSTEM_LIBS) -g
