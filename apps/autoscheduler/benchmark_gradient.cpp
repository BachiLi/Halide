#include "Halide.h"
#include "Derivative.h"
#include "DerivativeUtils.h"
#include "halide_benchmark.h"

#include <dlfcn.h>
#include <iostream>
#include <string>

using namespace Halide;
using namespace Halide::Tools;

struct PrintFuncOptions {
    bool ignore_non_adjoints = false;
    bool ignore_bc = false;
    int depth = -1;
    std::map<std::string, Expr> variables;
};

void print_func(const Func &func, const PrintFuncOptions &options = PrintFuncOptions{}) {
    Internal::debug(0) << "Printing function:" << func.name() << "\n";
    // Topologically sort the functions
    std::map<std::string, Internal::Function> env = find_transitive_calls(func.function());
    std::vector<std::string> order = realization_order({ func.function() }, env).first;
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        funcs.push_back(func);
    }

    int lowest_index = 0;
    if (options.depth >= 0) {
        lowest_index = (int) funcs.size() - 1 - options.depth;
    }

    for (int i = (int) funcs.size() - 1; i >= lowest_index; i--) {
        const char *ce = "constant_exterior";
        const char *re = "repeat_edge";
        if (options.ignore_bc && (funcs[i].name().substr(0, strlen(ce)) == std::string(ce) ||
                                  funcs[i].name().substr(0, strlen(re)) == std::string(re) ||
                                  funcs[i].name().find("_ce") != std::string::npos)) {
            continue;
        }
        if (options.ignore_non_adjoints && funcs[i].name().find("_d_def__") == std::string::npos) {
            continue;
        }
        Func func = funcs[i];
        Internal::debug(0) << "  funcs[" << i << "]: " << func.name() << "\n";
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            Internal::ReductionDomain rdom;
            if (update_id >= 0) {
                Internal::debug(0) << "    update:" << func.name() << "(";
                if (func.update_args(update_id).size() > 0) {
                    Expr e = func.update_args(update_id)[0];
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << Internal::simplify(e);
                    for (int i = 1; i < (int) func.update_args(update_id).size(); i++) {
                        Expr e = func.update_args(update_id)[i];
                        for (const auto &it : options.variables) {
                            e = substitute(it.first, it.second, e);
                        }
                        Internal::debug(0) << ", " << Internal::simplify(e);
                    }
                }
                Internal::debug(0) << ") =";
                auto vals = func.update_values(update_id).as_vector();
                for (auto val : vals) {
                    Expr e = val;
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << " " << Internal::simplify(e);
                }
                Internal::debug(0) << "\n";
                rdom = Internal::extract_rdom(Internal::simplify(func.update_value(update_id)));
            } else {
                Internal::debug(0) << "    " << func.name() << "(";
                if (func.args().size() > 0) {
                    Internal::debug(0) << func.args()[0];
                    for (int i = 1; i < (int) func.args().size(); i++) {
                        Internal::debug(0) << ", " << Internal::simplify(func.args()[i]);
                    }
                }
                Internal::debug(0) << ") =";
                auto vals = func.values().as_vector();
                for (auto val : vals) {
                    Expr e = val;
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << " " << Internal::simplify(e);
                }
                Internal::debug(0) << "\n";
                rdom = Internal::extract_rdom(Internal::simplify(func.value()));
            }

            if (rdom.defined()) {
                Internal::debug(0) << "    RDom:";
                for (int i = 0; i < (int) rdom.domain().size(); i++) {
                    Internal::debug(0) << " (" << Internal::simplify(rdom.domain()[i].min) << ", " << Internal::simplify(rdom.domain()[i].extent) << ")";
                }
                Internal::debug(0) << "\n";
            }
        }
    }
}

/*
void post_gradient_transform(std::vector<Func> &outputs) {
    std::map<std::string, Internal::Function> env;
    for (auto output : outputs) {
        std::map<std::string, Internal::Function> e = find_transitive_calls(output.function());
        for (auto it : e) {
            env[it->first] = it->second;
        }
    }
    std::vector<Internal::Function> output_functions;
    for (auto output : outputs) {
        output_functions.push_back(output.function());
    }
    std::vector<std::string> order = realization_order(output_functions, env).first;
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        funcs.push_back(func);
    }

    std::map<std::string, Box> bounds = inference_bounds(outputs);
    for (auto func : funcs) {
        // Search for f(
    }
}
*/

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
    int timing_iterations = 40;

    Var x("x"), y("y");

    for (int stage = 0; stage < 2; stage++) {
        if (stage == 0) {
            std::cout << "1D box filter without gradient" << std::endl;
        } else {
            std::cout << "1D box filter with gradient" << std::endl;
        }

        // Algorithm
        int n = 1048576;
        int w = 9;
        Buffer<float> in(n);
        for (int i = 0; i < n; i++) {
            in(i) = (float) i / (float) n;
        }
        Func in_clamped = BoundaryConditions::repeat_edge(in);
        Func f("f"), loss("loss");
        RDom r(0, w);
        f(x) += in_clamped(x - r);
        RDom r_f(0, n);
        loss() += f(r_f);

        auto d = propagate_adjoints(loss);
        Func d_in = d(in);

        f.estimate(x, 0, n);
        d_in.estimate(d_in.args()[0], 0, n);

        if (stage == 0) {
            Pipeline p = Pipeline({f});
            p.auto_schedule(target, params);
            Buffer<float> f_buffer(n);
            p.compile_jit(target);
            double best_time_fwd = benchmark(timing_iterations, 10, [&]() {
                p.realize(f_buffer);
            });
            std::cout << "forward best time:" << best_time_fwd << std::endl;
        } else {
            Pipeline p_grad = Pipeline({d_in});
            p_grad.auto_schedule(target, params);
            Buffer<float> d_in_buffer(n);
            p_grad.compile_jit(target);
            double best_time_grad = benchmark(timing_iterations, 10, [&]() {
                p_grad.realize(d_in_buffer);
            });
            std::cout << "gradient best time:" << best_time_grad << std::endl;
        }
    }

    for (int stage = 0; stage < 2; stage++) {
        if (stage == 0) {
            std::cout << "1D conv without gradient" << std::endl;
        } else {
            std::cout << "1D conv with gradient" << std::endl;
        }
        int n = 1048576;
        int w = 9;
        Buffer<float> in(n), k(w);
        for (int i = 0; i < n; i++) {
            in(i) = (float) i / (float) n;
        }
        for (int i = 0; i < w; i++) {
            k(i) = (float) i / (float) n;
        }
        Func in_clamped = BoundaryConditions::repeat_edge(in);
        Func k_inter("k"), f("f"), loss("loss");
        RDom r(0, w);
        k_inter(x, y) = k(x);
        f(x) += in_clamped(x - r) * k_inter(r, x);
        RDom r_f(0, n);
        loss() += f(r_f);

        auto d = propagate_adjoints(loss);
        Func d_in = d(in);
        Func d_k_inter = d(k_inter);

        f.estimate(x, 0, n);
        d_in.estimate(d_in.args()[0], 0, n);
        d_k_inter.estimate(x, 0, w);
        d_k_inter.estimate(y, 0, n);

        if (stage == 0) {
            Pipeline p = Pipeline({f});
            p.auto_schedule(target, params);
            Buffer<float> f_buffer(n);
            p.compile_jit(target);
            double best_time_fwd = benchmark(timing_iterations, 10, [&]() {
                p.realize(f_buffer);
            });
            std::cout << "forward best time:" << best_time_fwd << std::endl;
        } else {
            Pipeline p_grad = Pipeline({d_in, d_k_inter});
            p_grad.auto_schedule(target, params);
            Buffer<float> d_f_buffer(n), d_k_inter_buffer(w, n), d_k_buffer(w);
            // Manually scheduled reduction from k_inter to k
            Func d_k("d_k");
            d_k(x) += d_k_inter_buffer(x, r_f);
            RVar rxo, rxi, ryi;
            int tile_width = 32, tile_height = 32;
            d_k.update(0)
               .split(r_f, rxo, rxi, tile_width * tile_height, TailStrategy::GuardWithIf)
               .split(rxi, ryi, rxi, tile_width, TailStrategy::GuardWithIf);
            Var xo, yo, xi;
            // Parallel on tiles and vectorize inside tile
            Func k_reduction = d_k.update(0)
                                  .rfactor({{rxo, xo},
                                            {rxi, xi}});
            k_reduction.compute_root()
                .parallel(xo)
                .vectorize(xi);
            k_reduction.update()
                       .reorder({ryi, xi, x})
                       .parallel(xo)
                       .vectorize(xi);
            d_k.compute_root()
               .update();
            print_func(d_k);

            p_grad.compile_jit(target);
            d_k.compile_jit(target);
            double best_time_grad = benchmark(timing_iterations, 10, [&]() {
                p_grad.realize({d_f_buffer, d_k_inter_buffer});
                d_k.realize(d_k_buffer);
            });
            std::cout << "gradient best time:" << best_time_grad << std::endl;
        }
    }

    return 0;
}

