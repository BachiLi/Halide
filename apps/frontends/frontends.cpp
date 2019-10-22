#include "Halide.h"
#include "Lower.h"
#include <vector>

using namespace Halide;
using namespace Halide::Internal;
using std::vector;
using std::map;
using std::string;

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
    Expr x = Variable::make(Int(32), "x");
    s = Provide::make(f.name(), {x}, {x});
    Expr min = Variable::make(Int(32), out.name() + ".min.0");
    Expr extent = Variable::make(Int(32), out.name() + ".extent.0");
    s = For::make("x", min, extent, ForType::Serial, DeviceAPI::Host, s);
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f}, s);
    run(m, {}, {out});
    for (int i = 0; i < 16; i++) {
        _halide_user_assert(out(i) == i);
    }
}

int main(int argc, char *argv[]) {
    store_to_scalar();
    load_store_scalar();
    call_provide_scalar();
    provide_loop();
    return 0;
}
