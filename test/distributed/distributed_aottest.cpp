#include "HalideBuffer.h"
#include "HalideRuntime.h"

#include "distributed.h"

#include <mpi.h>
#include <iostream>

using namespace Halide::Runtime;

const int num_elements = 16383;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Buffer<int> input(num_elements);
    Buffer<int> output(nullptr, 0);
    input.set_distributed(std::vector<int>{num_elements});
    output.set_distributed(std::vector<int>{num_elements});
    distributed(input, output);
    input.allocate();
    output.allocate();

    for (int x = input.dim(0).min(); x <= input.dim(0).max(); x++) {
        input(x) = x;
    }
    distributed(input, output);

    for (int x = output.dim(0).min(); x <= output.dim(0).max(); x++) {
        assert(output(x) == 2 * input(x) + 1);
    }

    MPI_Finalize();
    if (rank == 0) {
        std::cout << "Success!" << std::endl;
    }
    return 0;
}
