#include "Halide.h"
#include "Errors.h"
#include "Lower.h"
#include <vector>
#include <functional>

using namespace Halide;
using namespace Halide::Internal;
using std::vector;
using std::map;
using std::string;
using std::pair;
using std::function;

Parameter parameter(const Buffer<> &buffer) {
    return Parameter(buffer.type(),
                     true /* is_buffer */,
                     buffer.dimensions(),
                     buffer.name());
}

JITModule compile(const vector<Buffer<>> &inputs,
                  const vector<Buffer<>> &outputs,
                  const vector<Parameter> &output_params,
                  Stmt s) {
    Target target = get_jit_target_from_environment();
    target.set_feature(Target::NoBoundsQuery, true);
    target.set_feature(Target::NoAsserts, true);
    target.set_feature(Target::JIT, true);

    map<string, Parameter> output_params_map;
    for (auto p : output_params) {
        output_params_map[p.name()] = p;
    }

    vector<Argument> arguments;
    for (auto b : inputs) {
        arguments.push_back(Argument(b));
    }
    for (auto b : outputs) {
        arguments.push_back(Argument(b));
    }
    const std::string &name = "foo";
    Module module = lower_from_stmt(s,
                                    name,
                                    output_params_map,
                                    target,
                                    arguments,
                                    LinkageType::ExternalPlusMetadata);
    auto func = module.get_function_by_name(name);
    return JITModule(module, func);
}

void run(JITModule m,
         const vector<Buffer<>> &inputs,
         const vector<Buffer<>> &outputs) {
    std::vector<const void*> raw_args;
    raw_args.reserve(inputs.size() + outputs.size());
    for (auto b : inputs) {
        raw_args.push_back((const void*)b.raw_buffer());
    }
    for (auto b : outputs) {
        raw_args.push_back((const void*)b.raw_buffer());
    }
    int exit = m.argv_function()(raw_args.data());
    _halide_user_assert(exit == 0);
}

/////////////////////////////////////////////////////////
// Helpers for constructing statements
Stmt ForAll(const vector<string> &names,
            const vector<pair<Expr, Expr>> &ranges,
            function<Stmt()> f,
            ForType for_type = ForType::Serial,
            DeviceAPI device_api = DeviceAPI::Host) {
    user_assert(names.size() == ranges.size());
    Stmt s = f();
    // Build loop nest
    for (int i = 0; i < (int)names.size(); i++) {
        s = For::make(names[i],
                      ranges[i].first,
                      ranges[i].second,
                      for_type,
                      device_api,
                      s);
    }
    return s;
}
/////////////////////////////////////////////////////////

/// Store 100 to a scalar function
void store_to_scalar() {
    Buffer<int> out = Buffer<int>::make_scalar("f");
    Parameter f = parameter(out);

    Stmt s;
    s = Store::make(f.name(), 100, 0, f, const_true(), ModulusRemainder());
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f}, s);
    run(m, {}, {out});
    _halide_user_assert(out() == 100);
}

/// Load from a scalar input, multiply by 2.
void load_store_scalar() {
    Buffer<int> in = Buffer<int>::make_scalar("f");
    Buffer<int> out = Buffer<int>::make_scalar("g");
    Parameter f = parameter(in);
    Parameter g = parameter(out);

    Stmt s;
    Expr e = Load::make(in.type(), in.name(), 0, in, f, const_true(), ModulusRemainder());
    e = 2 * e; // multiply by 2
    s = Store::make(g.name(), e, 0, g, const_true(), ModulusRemainder());
    s = ProducerConsumer::make_produce(g.name(), s);

    JITModule m = compile({in}, {out}, {g}, s);
    in() = 3;
    run(m, {in}, {out});
    _halide_user_assert(out() == 6);
}

/// Same as load_store_scalar, but with Call and Provide
void call_provide_scalar() {
    Buffer<int> in = Buffer<int>::make_scalar("f");
    Buffer<int> out = Buffer<int>::make_scalar("g");
    Parameter f = parameter(in);
    Parameter g = parameter(out);

    Stmt s;
    Expr e = Call::make(in, {});
    e = 2 * e; // multiply by 2
    s = Provide::make(g.name(), {e}, {});
    s = ProducerConsumer::make_produce(g.name(), s);

    JITModule m = compile({in}, {out}, {g}, s);
    in() = 3;
    run(m, {in}, {out});
    _halide_user_assert(out() == 6);
}

/// f(x) = x
void provide_loop() {
    Buffer<int> out = Buffer<int>(16, "f");
    Parameter f = parameter(out);

    Stmt s;
    Expr min = Variable::make(Int(32), out.name() + ".min.0");
    Expr extent = Variable::make(Int(32), out.name() + ".extent.0");
    s = ForAll({"x"}, {{min, extent}}, [&]() {
        Expr x = Variable::make(Int(32), "x");
        return Provide::make(f.name(), {x}, {x});
    });
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f}, s);
    run(m, {}, {out});
    for (int i = 0; i < 16; i++) {
        _halide_user_assert(out(i) == i);
    }
}

/// f(x, y) = x + y
void provide_loop_multidim() {
    Buffer<int> out = Buffer<int>(16, 8, "f");
    Parameter f = parameter(out);

    Stmt s;
    Expr x_min = Variable::make(Int(32), out.name() + ".min.0");
    Expr x_extent = Variable::make(Int(32), out.name() + ".extent.0");
    Expr y_min = Variable::make(Int(32), out.name() + ".min.1");
    Expr y_extent = Variable::make(Int(32), out.name() + ".extent.1");    
    s = ForAll({"x", "y"}, {{x_min, x_extent}, {y_min, y_extent}}, [&]() {
        Expr x = Variable::make(Int(32), "x");
        Expr y = Variable::make(Int(32), "y");
        return Provide::make(f.name(), {x + y}, {x, y});
    });
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f}, s);
    run(m, {}, {out});
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            _halide_user_assert(out(i, j) == i + j);
        }
    }
}

/// g(x, y) = 2 * f(x, y)
void call_provide_loop_multidim() {
    Buffer<int> in = Buffer<int>(16, 8, "f");
    Buffer<int> out = Buffer<int>(16, 8, "g");
    Parameter f = parameter(in);
    Parameter g = parameter(out);

    Stmt s;
    Expr x_min = Variable::make(Int(32), out.name() + ".min.0");
    Expr x_extent = Variable::make(Int(32), out.name() + ".extent.0");
    Expr y_min = Variable::make(Int(32), out.name() + ".min.1");
    Expr y_extent = Variable::make(Int(32), out.name() + ".extent.1");
    s = ForAll({"x", "y"}, {{x_min, x_extent}, {y_min, y_extent}}, [&]() {
        Expr x = Variable::make(Int(32), "x");
        Expr y = Variable::make(Int(32), "y");
        Expr e = 2 * Call::make(in, {x, y});
        return Provide::make(g.name(), {e}, {x, y});
    });
    s = ProducerConsumer::make_produce(g.name(), s);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            in(i, j) = i + j;
        }
    }
    JITModule m = compile({in}, {out}, {g}, s);
    run(m, {in}, {out});
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            _halide_user_assert(out(i, j) == in(i, j) * 2);
        }
    }
}

void in_place_bubble_sort() {
    Buffer<int> in_out = Buffer<int>(16, "f");
    Parameter f = parameter(in_out);

    // for (int i = 0; i < size; i++) {
    //     for (int j = 1; j < size; j++) {
    //         if (f(j - 1) > f(j)) {
    //             Allocate(_t);
    //             _t = f(j - 1);
    //             f(j - 1) = f(j);
    //             f(j) = _t;
    //         }
    //     }
    // }
    Stmt s;
    Expr f_min = Variable::make(Int(32), in_out.name() + ".min.0");
    Expr f_extent = Variable::make(Int(32), in_out.name() + ".extent.0");
    s = ForAll({"j", "i"}, {{f_min + 1, f_extent - 1}, {f_min, f_extent}}, [&]() {
        Expr i = Variable::make(Int(32), "i");
        Expr j = Variable::make(Int(32), "j");
        Expr prev = Call::make(in_out, {j - 1});
        Expr curr = Call::make(in_out, {j});
        // _t = f(j - 1)
        Stmt s = Store::make("_t", prev, 0, Parameter(), const_true(), ModulusRemainder());
        // f(j - 1) = f(j)
        s = Block::make(s, Provide::make(f.name(), {curr}, {j - 1}));
        // f(j) = _t
        Expr _t = Load::make(Int(32), "_t", 0, Buffer<>(), Parameter(), const_true(), ModulusRemainder());
        s = Block::make(s, Provide::make(f.name(), {_t}, {j}));
        // Allocate(_t)
        s = Allocate::make("_t", Int(32), MemoryType::Register, {}, const_true(), s);
        // if (f(j - 1) > f(j))
        s = IfThenElse::make(prev > curr, s);
        return s;
    });
    s = ProducerConsumer::make_produce(f.name(), s);

    for (int i = 0; i < 16; i++) {
        in_out(i) = 15 - i;
    }
    JITModule m = compile({in_out}, {in_out}, {f}, s);
    run(m, {in_out}, {in_out});
    for (int i = 0; i < 16; i++) {
        _halide_user_assert(in_out(i) == i);
    }
}

int main(int argc, char *argv[]) {
    store_to_scalar();
    load_store_scalar();
    call_provide_scalar();
    provide_loop();
    provide_loop_multidim();
    call_provide_loop_multidim();
    in_place_bubble_sort();
    return 0;
}
