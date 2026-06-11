.PHONY: all clean

# --- Compiler ---
CXX := g++
# To use Intel: CXX := icpc

# --- Eigen include path (adjust to your system) ---
EIGEN_DIR := /data/finite/sazizi/new_non_adiabatic/forThesisi

# --- Flags ---
CXXFLAGS := -std=c++17 -Wall -O3 -DNDEBUG -funroll-loops -march=native -fopenmp
INCLUDES := -I$(EIGEN_DIR)
LDFLAGS  := -fopenmp
LDLIBS   := -lgsl -lgslcblas -lm

BUILDDIR := ./build/

$(shell mkdir -p $(BUILDDIR) basissets)

SRCS := $(wildcard *.cpp)
OBJS := $(addprefix $(BUILDDIR), $(SRCS:.cpp=.o))
DEPS := $(OBJS:.o=.d)

all: main

main: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILDDIR)%.o: %.cpp
	@echo ">> Compiling $< ..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -rf $(BUILDDIR) main
