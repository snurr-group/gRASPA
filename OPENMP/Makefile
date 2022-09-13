
CXX=nvc++

CXXFLAGS=-O3 -std=c++20 -mp=gpu -Minfo -g 

main.x: main.o read_data.o convert_array.o GPU_alloc.o framework_energy_difference.o
	$(CXX) $(CXXFLAGS) main.o read_data.o convert_array.o GPU_alloc.o framework_energy_difference.o -o main.x
main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp
convert_array.o: convert_array.cpp
	$(CXX) $(CXXFLAGS) -c convert_array.cpp
framework_energy_difference.o: framework_energy_difference.cpp
	$(CXX) $(CXXFLAGS) -c framework_energy_difference.cpp
GPU_alloc.o: GPU_alloc.cpp
	$(CXX) $(CXXFLAGS) -c GPU_alloc.cpp
read_data.o: read_data.cpp
	$(CXX) $(CXXFLAGS) -c read_data.cpp
clean:
	rm *.o *.x 

