#include "Halide.h"
#include "Derivative.h"
#include "halide_benchmark.h"

#include <dlfcn.h>
#include <iostream>
#include <string>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    std::string use_master_autoscheduler =
        Halide::Internal::get_env_variable("HL_USE_MASTER_AUTOSCHEDULER");
    if (use_master_autoscheduler.empty()) { 
        std::cout << "Use new autoscheduler" << std::endl;
        if (!dlopen("libauto_schedule.so", RTLD_LAZY)) {
            std::cerr << "Failed to load autoscheduler: " << dlerror() << "\n";
            return 1;
        }
    } else {
        std::cout << "Use master autoscheduler" << std::endl;
        Pipeline::set_custom_auto_scheduler(nullptr);
    }

    MachineParams params(32, 16000000, 40);
    // Use a fixed target for the analysis to get consistent results from this test.
    Target target("x86-64-linux-sse41-avx-avx2");
    int timing_iterations = 10;

    Var x("x"), y("y");

    if (1) {
        // For time calibration
        std::cout << "1D box filter without gradient" << std::endl;
        int n = 1000000;
        Func f("f"), g("g"), h("h");
        RDom r(0, 5);
        f(x) = cast<float>(x);
        g(x) += f(x - r);

        g.estimate(x, 0, n);

        Pipeline p = Pipeline({g});
        p.auto_schedule(target, params);
        Buffer<float> g_buffer(n);
        p.compile_jit(target);
        double best_time = benchmark(timing_iterations, 10, [&]() {
            p.realize(g_buffer);
        });
        std::cout << "best time:" << best_time << std::endl;
    }


    if (1) {
        std::cout << "1D box filter with gradient" << std::endl;
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

    if (1) {
        std::cout << "1D conv with gradient" << std::endl;
        int n = 1000000;
        Func f("f"), g("g"), h("h"), k("k");
        RDom r(0, 5);
        f(x) = cast<float>(x);
        k(x) = cast<float>(x);
        g(x) += f(x - r) * k(r);
        RDom r_f(0, n);
        h() += g(r_f);

        auto d = propagate_adjoints(h);
        Func d_f = d(f);
        Func d_k = d(k);

        d_f.estimate(x, 0, n);
        d_k.estimate(x, 0, 5);

        Pipeline p = Pipeline({d_f, d_k});
        p.auto_schedule(target, params);
        Buffer<float> d_f_buffer(n), d_k_buffer(5);
        p.compile_jit(target);
        double best_time = benchmark(timing_iterations, 10, [&]() {
            p.realize({d_f_buffer, d_k_buffer});
        });
        std::cout << "best time:" << best_time << std::endl;
    }

    return 0;
}
