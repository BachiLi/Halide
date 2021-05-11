#include "Halide.h"
#include <mpi.h>
#include <iomanip>
#include <stdarg.h>
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Var x, y, z, c;
    Var xo, yo, xi, yi, tile;
    {
        Func f;
        f(x) = 2 * x + 1;
        f.distribute(x);

        Buffer<int> out = f.realize({20});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 2 * x + 1;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    { // same test as above, but with a non-divisible extent
        Func f;
        f(x) = 2 * x + 1;
        f.distribute(x);
        Buffer<int> out = f.realize({25});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 2 * x + 1;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = 2 * in(x) + 1;
        f.distribute(x);

        int num_elements = 23;
        f.infer_input_bounds({num_elements});
        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, num_elements - 1);
        Buffer<int> in_buf = in.get();
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }

        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }
        Buffer<int> out = f.realize({num_elements});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 1;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = 2 * in(x) + 1;
        Func g;
        g(x) = f(x) + 1;
        g.distribute(x);
        f.compute_root().distribute(x);

        int num_elements = 23;
        g.infer_input_bounds({num_elements});
        Buffer<int> in_buf = in.get();
        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, num_elements - 1);
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }

        Buffer<int> out = g.realize({num_elements});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 2;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = 2 * in(x) + 1;
        Func g;
        g(x) = f(x) + 1;
        g.distribute(x);
        f.compute_root();

        int num_elements = 23;
        g.infer_input_bounds({num_elements});
        Buffer<int> in_buf = in.get();
        int buf_min = 0;
        int buf_max = num_elements - 1;
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }

        Buffer<int> out = g.realize({num_elements});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 2;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        int num_elements = 23;
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = in(max(x - 1, 0)) + in(x) + in(min(x + 1, in.global_width() - 1));
        f.distribute(x);

        Buffer<int> in_buf(nullptr, 0);
        in_buf.set_distributed(std::vector<int>{num_elements});
        f.infer_input_bounds({num_elements}, get_jit_target_from_environment(), {{in, &in_buf}});
        in.set(in_buf);

        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = std::max(num_elements_per_proc * rank - 1, 0);
        int buf_max = std::min((num_elements_per_proc * rank - 1) + num_elements_per_proc + 1, num_elements - 1);
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }

        Buffer<int> out = f.realize({num_elements});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int left_id = std::max(x - 1, 0);
            int right_id = std::min(x + 1, num_elements - 1);
            int correct = 2 * left_id + 2 * x + 2 * right_id;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        int num_elements = 23;
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = in(max(x - 1, 0)) + in(x) + in(min(x + 1, in.global_width() - 1));
        f.distribute(x);

        Buffer<int> in_buf(nullptr, 0);
        in_buf.set_distributed(std::vector<int>{num_elements});
        f.infer_input_bounds({num_elements}, get_jit_target_from_environment(), {{in, &in_buf}});
        in.set(in_buf);

        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = std::max(num_elements_per_proc * rank - 1, 0);
        int buf_max = std::min((num_elements_per_proc * rank - 1) + num_elements_per_proc + 1, num_elements - 1);
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }

        Buffer<int> out = f.realize({num_elements});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int left_id = std::max(x - 1, 0);
            int right_id = std::min(x + 1, num_elements - 1);
            int correct = 2 * left_id + 2 * x + 2 * right_id;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = 2 * in(x) + 1;
        f(x) = f(x) + 1;
        f.distribute(x);
        f.update().distribute(x);

        int num_elements = 23;
        f.infer_input_bounds({num_elements});
        Buffer<int> in_buf = in.get();
        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, num_elements - 1);
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }

        Buffer<int> out = f.realize({num_elements});
        if (out.dim(0).min() != buf_min) {
            printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (out.dim(0).max() != buf_max) {
            printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 2;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        ImageParam in(Int(32), 1, "in");
        Func f("f");
        f(x) = 2 * in(x) + 1;
        Var xo, xi;
        f.compute_root()
         .split(x, xo, xi, 64);
        f.distribute(xo);

        int num_elements = 16383;
        Buffer<int> in_buf(nullptr, 0);
        in_buf.set_distributed(std::vector<int>{num_elements});
        f.infer_input_bounds({num_elements}, get_jit_target_from_environment(), {{in, &in_buf}});
        in.set(in_buf);
        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, num_elements - 1);
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }

        Buffer<int> out = f.realize({num_elements});
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 1;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        Func f("f");
        f(x, y) = 2 * x + 2 * y + 1;
        f.distribute(y);
        Buffer<int> out = f.realize({20, 20});
        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (out.dim(0).min() != 0) {
            printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
            return 0;
        }
        if (out.dim(0).max() != 19) {
            printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 19);
            return 0;
        }
        if (out.dim(1).min() != buf_min) {
            printf("rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), buf_min);
            return 0;
        }
        if (out.dim(1).max() != buf_max) {
            printf("rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), buf_max);
            return 0;
        }
        for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
            for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                int correct = 2 * x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        Func f("f");
        f(x) = 2 * x + 1;
        f.distribute(x)
         .send_to(0);
        Buffer<int> out = f.realize({20});
        if (rank == 0) {
            if (out.dim(0).min() != 0) {
                printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
                return 0;
            }
            if (out.dim(0).max() != 19) {
                printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 19);
                return 0;
            }
        } else {
            int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
            int buf_min = num_elements_per_proc * rank;
            int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
            if (out.dim(0).min() != buf_min) {
                printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), buf_min);
                return 0;
            }
            if (out.dim(0).max() != buf_max) {
                printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), buf_max);
                return 0;
            }
        }
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 2 * x + 1;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        Func f("f"), g("g");
        f(x) = 2 * x + 1;
        g(x) = 2 * f(x) + 1;
        f.compute_root()
         .distribute(x);
        g.distribute(x)
         .send_to(0);
        Buffer<int> out = g.realize({20});
        if (rank == 0) {
            if (out.dim(0).min() != 0) {
                printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
                return 0;
            }
            if (out.dim(0).max() != 19) {
                printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 19);
                return 0;
            }
        } else {
            int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
            int buf_min = num_elements_per_proc * rank;
            int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
            if (out.dim(0).min() != buf_min) {
                printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), buf_min);
                return 0;
            }
            if (out.dim(0).max() != buf_max) {
                printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), buf_max);
                return 0;
            }
        }
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 3;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        Func f("f");
        f(x, y) = 2 * x + 2 * y + 1;
        f.distribute(y)
         .send_to(0);
        Buffer<int> out = f.realize({10, 20});
        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (rank == 0) {
            buf_min = 0;
            buf_max = 19;
        }
        if (out.dim(0).min() != 0) {
            printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
            return 0;
        }
        if (out.dim(0).max() != 9) {
            printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 9);
            return 0;
        }
        if (out.dim(1).min() != buf_min) {
            printf("rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), buf_min);
            return 0;
        }
        if (out.dim(1).max() != buf_max) {
            printf("rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), buf_max);
            return 0;
        }
        for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
            for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                int correct = 2 * x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        Func f("f");
        f(x, y, z) = 2 * x + 3 * y + 5 * z + 1;
        f.distribute(z)
         .send_to(0);
        Buffer<int> out = f.realize({5, 7, 20});
        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (rank == 0) {
            buf_min = 0;
            buf_max = 19;
        }
        if (out.dim(0).min() != 0) {
            printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
            return 0;
        }
        if (out.dim(0).max() != 4) {
            printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 4);
            return 0;
        }
        if (out.dim(1).min() != 0) {
            printf("rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), 0);
            return 0;
        }
        if (out.dim(1).max() != 6) {
            printf("rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), 6);
            return 0;
        }
        if (out.dim(2).min() != buf_min) {
            printf("rank %d: out.dim(2).min() = %d instead of %d\n", rank, out.dim(2).min(), buf_min);
            return 0;
        }
        if (out.dim(2).max() != buf_max) {
            printf("rank %d: out.dim(2).max() = %d instead of %d\n", rank, out.dim(2).max(), buf_max);
            return 0;
        }
        for (int z = out.dim(2).min(); z <= out.dim(2).max(); z++) {
            for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
                for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                    int correct = 2 * x + 3 * y + 5 * z + 1;
                    if (out(x, y, z) != correct) {
                        printf("out(%d, %d, %d) = %d instead of %d\n", x, y, z, out(x, y, z), correct);
                        MPI_Finalize();
                        return -1;
                    }
                }
            }
        }
    }

    if (get_jit_target_from_environment().has_gpu_feature()) {
        Func f("f");
        f(x, y, z) = 2 * x + 3 * y + 5 * z + 1;
        f.distribute(z)
         .gpu_blocks(y)
         .gpu_threads(x)
         .send_to(0);
        Buffer<int> out = f.realize({5, 7, 20});
        out.copy_to_host();

        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (rank == 0) {
            buf_min = 0;
            buf_max = 19;
        }
        if (out.dim(0).min() != 0) {
            printf("rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
            return 0;
        }
        if (out.dim(0).max() != 4) {
            printf("rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 4);
            return 0;
        }
        if (out.dim(1).min() != 0) {
            printf("rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), 0);
            return 0;
        }
        if (out.dim(1).max() != 6) {
            printf("rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), 6);
            return 0;
        }
        if (out.dim(2).min() != buf_min) {
            printf("rank %d: out.dim(2).min() = %d instead of %d\n", rank, out.dim(2).min(), buf_min);
            return 0;
        }
        if (out.dim(2).max() != buf_max) {
            printf("rank %d: out.dim(2).max() = %d instead of %d\n", rank, out.dim(2).max(), buf_max);
            return 0;
        }
        for (int z = out.dim(2).min(); z <= out.dim(2).max(); z++) {
            for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
                for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                    int correct = 2 * x + 3 * y + 5 * z + 1;
                    if (out(x, y, z) != correct) {
                        printf("out(%d, %d, %d) = %d instead of %d\n", x, y, z, out(x, y, z), correct);
                        MPI_Finalize();
                        return -1;
                    }
                }
            }
        }
    }

    {
        // split vertically over tiles
        Func f("f");
        f(x, y) = 2 * x + 2 * y + 1;
        f.tile({x, y}, {xo, yo}, {xi, yi}, {10, 10})
         .fuse(xo, yo, tile)
         .distribute(tile)
         .send_to(0);

        Buffer<int> out = f.realize({10, 20});
        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (rank == 0) {
            buf_min = 0;
            buf_max = 19;
        }
        if (out.dim(0).min() != 0) {
            printf("split-vertical-tiles: rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), 0);
            return -1;
        }
        if (out.dim(0).max() != 9) {
            printf("split-vertical-tiles: rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), 9);
            return -1;
        }
        if (out.dim(1).min() != buf_min) {
            printf("split-vertical-tiles: rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), buf_min);
            return -1;
        }
        if (out.dim(1).max() != buf_max) {
            printf("split-vertical-tiles: rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), buf_max);
            return -1;
        }
        for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
            for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                int correct = 2 * x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        // split horizontally over tiles
        Func f("f");
        f(x, y) = 2 * x + 2 * y + 1;
        f.reorder(y, x)
         .tile({x, y}, {xo, yo}, {xi, yi}, {10, 10})
         .fuse(xo, yo, tile)
         .distribute(tile)
         .send_to(0);

        Buffer<int> out = f.realize({20, 10});
        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (rank == 0) {
            buf_min = 0;
            buf_max = 19;
        }
        if (out.dim(0).min() != buf_min) {
            printf("split-horizontal-tiles: rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), buf_min);
            return -1;
        }
        if (out.dim(0).max() != buf_max) {
            printf("split-horizontal-tiles: rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), buf_max);
            return -1;
        }
        if (out.dim(1).min() != 0) {
            printf("split-horizontal-tiles: rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), 0);
            return -1;
        }
        if (out.dim(1).max() != 9) {
            printf("split-horizontal-tiles: rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), 9);
            return -1;
        }
        for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
            for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                int correct = 2 * x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        // split unevenly over tiles
        Func f("f");
        f(x, y) = 2 * x + 2 * y + 1;
        f.tile({x, y}, {xo, yo}, {xi, yi}, {10, 10})
         .fuse(xo, yo, tile)
         .distribute(tile)
         .send_to(0);

        Buffer<int> out = f.realize({30, 30});
        for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
            for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                int correct = 2 * x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        // check for problems using bound + distribute
        Func f("f");
        f(x, y) = 2 * x + 2 * y + 1;
        f.bound(x, 0, 10)
         .bound(y, 0, 20)
         .distribute(y)
         .send_to(0);

        Buffer<int> out = f.realize({10, 20});
        int num_elements_per_proc = (20 + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, 20 - 1);
        if (rank == 0) {
            buf_min = 0;
            buf_max = 19;
        }
        if (out.dim(0).min() != buf_min) {
            printf("bound+distribute: rank %d: out.dim(0).min() = %d instead of %d\n", rank, out.dim(0).min(), buf_min);
            return -1;
        }
        if (out.dim(0).max() != buf_max) {
            printf("bound+distribute: rank %d: out.dim(0).max() = %d instead of %d\n", rank, out.dim(0).max(), buf_max);
            return -1;
        }
        if (out.dim(1).min() != 0) {
            printf("bound+distribute: rank %d: out.dim(1).min() = %d instead of %d\n", rank, out.dim(1).min(), 0);
            return -1;
        }
        if (out.dim(1).max() != 9) {
            printf("bound+distribute: rank %d: out.dim(1).max() = %d instead of %d\n", rank, out.dim(1).max(), 9);
            return -1;
        }
        for (int y = out.dim(1).min(); y <= out.dim(1).max(); y++) {
            for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
                int correct = 2 * x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
