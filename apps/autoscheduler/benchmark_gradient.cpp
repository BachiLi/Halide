#include "Halide.h"
#include "Derivative.h"
#include "halide_benchmark.h"

#include <dlfcn.h>
#include <iostream>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    if (!dlopen("libauto_schedule.so", RTLD_LAZY)) {
        std::cerr << "Failed to load autoscheduler: " << dlerror() << "\n";
        return 1;
    }

    MachineParams params(32, 16000000, 40);
    // Use a fixed target for the analysis to get consistent results from this test.
    Target target("x86-64-linux-sse41-avx-avx2");
    int timing_iterations = 20;

    Var x("x"), y("y");

    if (1) {
        std::cout << "1D box filter" << std::endl;
        int n = 1000000;
        Func f("f"), g("g"), h("h");
        RDom r(0, 5);
        f(x) = cast<float>(x);
        g(x) += f(x - r);
        RDom r_f(0, n);
        h() += g(r_f);

        auto d = propagate_adjoints(h);
        Func d_f = d(f);

        d_f.estimate(x, 0, n);

        Pipeline p = Pipeline({d_f});
        p.auto_schedule(target, params);
        Buffer<float> d_f_buffer(n);
        p.compile_jit(target);
        double best_time = benchmark(timing_iterations, 10, [&]() {
            p.realize(d_f_buffer);
        });
        std::cout << "best time:" << best_time << std::endl;
    }

    return 0;
}
