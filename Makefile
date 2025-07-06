CXX = g++
CXXFLAGS = -std=c++11 -I/opt/homebrew/include/eigen3
LDFLAGS = -L/opt/homebrew/lib

FDFD_SRC = src/fdfd.cpp
FDTD_SRC = src/fdtd.cpp
FDFD_OBJ = $(FDFD_SRC:.cpp=.o)
FDTD_OBJ = $(FDTD_SRC:.cpp=.o)
FDFD_TARGET = target/fdfd
FDTD_TARGET = target/fdtd

all: $(FDFD_TARGET) $(FDTD_TARGET)

$(FDFD_TARGET): $(FDFD_OBJ)
	@mkdir -p target
	$(CXX) $(FDFD_OBJ) $(LDFLAGS) -o $(FDFD_TARGET)

$(FDTD_TARGET): $(FDTD_OBJ)
	@mkdir -p target
	$(CXX) $(FDTD_OBJ) $(LDFLAGS) -o $(FDTD_TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(FDFD_OBJ) $(FDTD_OBJ) $(FDFD_TARGET) $(FDTD_TARGET)
