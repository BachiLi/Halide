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
using std::to_string;

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
struct LoopVar {
    LoopVar(const string &name, Expr min, Expr extent)
        : name(name), min(min), extent(extent) {}

    operator Expr() const {
        return Variable::make(Int(32), name);
    }

    string name;
    Expr min, extent;
};

struct Vectorize {
    Vectorize(Expr factor = 1) : factor(factor) {}
    Expr factor;
};

struct Parallelize {
    Parallelize(Expr num_tasks = 1) : num_tasks(num_tasks) {}
    Expr num_tasks;
};

Stmt Split_(Stmt s,
            Expr size,
            bool size_is_inner,
            ForType outer_for_type,
            ForType inner_for_type) {
    const For *for_stmt = s.as<For>();
    user_assert(for_stmt) << "Can only split statement with an outer loop.";
    const string &loop_var_name = for_stmt->name;
    Stmt s_ = for_stmt->body;

    Expr inner_size, outer_size;
    if (size_is_inner) {
        inner_size = size;
        outer_size = select(for_stmt->extent % inner_size != 0,
                            for_stmt->extent / inner_size + 1,
                            for_stmt->extent / inner_size);
    } else {
        outer_size = size;
        inner_size = select(for_stmt->extent % outer_size != 0,
                            for_stmt->extent / outer_size + 1,
                            for_stmt->extent / outer_size);
    }

    Expr loop_var = Variable::make(Int(32), loop_var_name);
    Expr outer = loop_var;
    const string &inner_name = unique_name('v');
    Expr inner = Variable::make(outer.type(), inner_name);
    Expr rebased = outer * inner_size + inner;
    // Substitute variable var with rebased
    s_ = substitute(loop_var_name, rebased, s_);
    if (!can_prove(for_stmt->extent % size == 0)) {
        // GuardWithIf
        s_ = IfThenElse::make(
            likely(rebased < for_stmt->min + for_stmt->extent), s_);
    }
    s_ = For::make(inner_name,
                   0,
                   inner_size,
                   inner_for_type,
                   DeviceAPI::Host,
                   s_);
    s_ = For::make(loop_var_name,
                   for_stmt->min,
                   outer_size,
                   outer_for_type,
                   DeviceAPI::Host,
                   s_);
    return s_;
}

Stmt ForAll(const vector<LoopVar> &vars,
            Stmt s,
            Vectorize v = Vectorize(1),
            Parallelize p = Parallelize(1)) {
    int vectorize_factor = 1;
    if (as_const_int(v.factor) != nullptr) {
        vectorize_factor = *as_const_int(v.factor);
    }
    // Build loop nest
    for (int var_id = 0; var_id < (int)vars.size(); var_id++) {
        const LoopVar &var = vars[var_id];
        s = For::make(var.name,
                      var.min,
                      var.extent,
                      ForType::Serial,
                      DeviceAPI::Host,
                      s);
        if (var_id == 0 && vectorize_factor > 1) {
            s = Split_(s, v.factor, true, ForType::Serial, ForType::Vectorized);
        }
        if (var_id == (int)vars.size() - 1 && !is_const(p.num_tasks, 1)) {
            s = Split_(s, p.num_tasks, false, ForType::Parallel, ForType::Serial);
        }
    }
    return s;
}

Stmt ForAll(const LoopVar &var,
            Stmt s,
            Vectorize v = Vectorize(1),
            Parallelize p = Parallelize(1)) {
    return ForAll(vector<LoopVar>{var}, s, v, p);
}

Stmt If(Expr condition,
        Stmt s) {
    return IfThenElse::make(condition, s);
}

struct InOutBufferRef;

struct InOutBuffer {
    InOutBuffer(const Buffer<> &b) : p(parameter(b)) {} 

    InOutBufferRef operator()(const vector<Expr> &args = {}) const;
    template<typename... Args>
    InOutBufferRef operator()(Expr x, Args &&... args) const;

    const string &name() const {
        return p.name();
    }

    Parameter param() const {
        return p;
    }

    Expr min(int index = 0) const {
        return Variable::make(Int(32), name() + ".min." + to_string(index));
    }

    Expr extent(int index = 0) const {
        return Variable::make(Int(32), name() + ".extent." + to_string(index));
    }

    Parameter p;
};

struct InOutBufferRef {
    InOutBufferRef(const InOutBuffer &b, const vector<Expr> &args)
        : b(b), args(args) {}

    Stmt operator=(Expr value) {
        return Provide::make(b.p.name(), {value}, args);
    }

    Stmt operator=(const InOutBufferRef &b) {
        return *this = Expr(b);
    }

    operator Expr() const {
        return Call::make(b.p, args);
    }

    const InOutBuffer &b;
    vector<Expr> args;
};

InOutBufferRef InOutBuffer::operator()(const vector<Expr> &args) const {
    return InOutBufferRef(*this, args);
}

template<typename... Args>
InOutBufferRef InOutBuffer::operator()(Expr x, Args &&... args) const {
    vector<Expr> collected_args{x, std::forward<Args>(args)...};
    return this->operator()(collected_args);
}

struct TempVar {
    TempVar(Type t, const string &name = unique_name('t'))
        : t(t), name(name) {}

    Stmt operator=(Expr e) {
        return Store::make(name, e, 0, Parameter(), const_true(), ModulusRemainder());
    }

    operator Expr() const {
        return Load::make(t, name, 0, Buffer<>(), Parameter(), const_true(), ModulusRemainder());
    }

    Type t;
    string name;
};

Stmt Scope_(TempVar t, Stmt s) {
    s = Allocate::make(t.name, Int(32), MemoryType::Register, {}, const_true(), s);
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
    InOutBuffer f(in);
    InOutBuffer g(out);

    Stmt s;
    s = (g() = 2 * f());
    s = ProducerConsumer::make_produce(g.name(), s);

    JITModule m = compile({in}, {out}, {g.param()}, s);
    in() = 3;
    run(m, {in}, {out});
    _halide_user_assert(out() == 6);
}

/// f(x) = x
void provide_loop() {
    Buffer<int> out = Buffer<int>(16, "f");
    InOutBuffer f(out);

    Stmt s;
    LoopVar x("x", f.min(), f.extent());
    s = ForAll(x,
        f(x) = x
    );
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f.param()}, s);
    run(m, {}, {out});
    for (int i = 0; i < 16; i++) {
        _halide_user_assert(out(i) == i);
    }
}

/// f(x, y) = x + y
void provide_loop_multidim() {
    Buffer<int> out = Buffer<int>(16, 8, "f");
    InOutBuffer f(out);

    Stmt s;
    LoopVar x("x", f.min(0), f.extent(0));
    LoopVar y("y", f.min(1), f.extent(1));
    s = ForAll({x, y},
        f(x, y) = x + y
    );
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f.param()}, s);
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
    InOutBuffer f(in);
    InOutBuffer g(out);

    Stmt s;
    LoopVar x("x", g.min(0), g.extent(0));
    LoopVar y("y", g.min(1), g.extent(1));
    s = ForAll({x, y}, 
        g(x, y) = 2 * f(x, y)
    );
    s = ProducerConsumer::make_produce(g.name(), s);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            in(i, j) = i + j;
        }
    }
    JITModule m = compile({in}, {out}, {g.param()}, s);
    run(m, {in}, {out});
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            _halide_user_assert(out(i, j) == in(i, j) * 2);
        }
    }
}

void in_place_bubble_sort() {
    Buffer<int> in_out = Buffer<int>(16, "f");
    InOutBuffer f(in_out);

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
    LoopVar j("j", f.min() + 1, f.extent() - 1);
    LoopVar i("i", f.min(), f.extent());
    TempVar t(Int(32));
    s = ForAll({j, i},
        If(f(j - 1) > f(j),
            Scope_(t, Block::make({
                t = f(j - 1),
                f(j - 1) = f(j),
                f(j) = t
            }))
        )
    );
    s = ProducerConsumer::make_produce(f.name(), s);

    for (int i = 0; i < 16; i++) {
        in_out(i) = 15 - i;
    }
    JITModule m = compile({in_out}, {in_out}, {f.param()}, s);
    run(m, {in_out}, {in_out});
    for (int i = 0; i < 16; i++) {
        _halide_user_assert(in_out(i) == i);
    }
}

/// h(x) = f(x) + g(x)
void add_vectorize() {
    Buffer<int> in0(17, "f");
    Buffer<int> in1(17, "g");
    Buffer<int> out(17, "h");
    InOutBuffer f(in0);
    InOutBuffer g(in1);
    InOutBuffer h(out);

    Stmt s;
    LoopVar x("x", h.min(), h.extent());
    s = ForAll(x,
        h(x) = f(x) + g(x),
        Vectorize(8));
    s = ProducerConsumer::make_produce(h.name(), s);

    for (int i = 0; i < 17; i++) {
        in0(i) = i;
        in1(i) = 17 - i;
    }
    JITModule m = compile({in0, in1}, {out}, {h.param()}, s);
    run(m, {in0, in1}, {out});
    for (int i = 0; i < 17; i++) {
        _halide_user_assert(out(i) == 17);
    }
}

/// h(x) = f(x) + g(x)
void add_parallelize_vectorize() {
    Buffer<int> in0(129, "f");
    Buffer<int> in1(129, "g");
    Buffer<int> out(129, "h");
    InOutBuffer f(in0);
    InOutBuffer g(in1);
    InOutBuffer h(out);

    Stmt s;
    LoopVar x("x", h.min(), h.extent());
    s = ForAll(x,
        h(x) = f(x) + g(x),
        Vectorize(8),
        Parallelize(8));
    s = ProducerConsumer::make_produce(h.name(), s);

    for (int i = 0; i < 129; i++) {
        in0(i) = i;
        in1(i) = 129 - i;
    }
    JITModule m = compile({in0, in1}, {out}, {h.param()}, s);
    run(m, {in0, in1}, {out});
    for (int i = 0; i < 129; i++) {
        _halide_user_assert(out(i) == 129);
    }
}

/// f(x) = 0
/// f(x) = x if x < 10
void test_break() {
    Buffer<int> out(15, "h");
    InOutBuffer f(out);

    Stmt s;
    LoopVar x("x", f.min(), f.extent());
    s = ForAll(x, f(x) = 0);
    s = Block::make({s,
        ForAll(x, Block::make({
        If(x >= 10, Break::make(x.name)),
        f(x) = x,
    }))});
    s = ProducerConsumer::make_produce(f.name(), s);

    JITModule m = compile({}, {out}, {f.param()}, s);
    run(m, {}, {out});
    for (int i = 0; i < 10; i++) {
        _halide_user_assert(out(i) == i);
    }
    for (int i = 11; i < 15; i++) {
        _halide_user_assert(out(i) == 0);
    }
}

// void mandelbrot() {
//     Buffer<int> out(512, 512, "h");
//     InOutBuffer f(out);

//     Stmt s;
//     LoopVar x("x", f.min(0), f.extent(0));
//     LoopVar y("y", f.min(1), f.extent(1));
//     LoopVar i("i", 0, 512);
//     TempVar z_re(Float(32)), z_im(Float(32)), count(Int(32));
//     s = ForAll({x, y}, Block::make({
//             z_re = cast<float>(x),
//             z_im = cast<float>(y),
//             ForAll(i, Block::make({
//                 If(z_re * z_re + z_im * z_im > 4.f, Block::make({
//                     count = i,
//                     Break::make()
//                 })), Block::make({
//                     z_re = cast<float>(x) + z_re * z_re - z_im * z_im,
//                     z_im = cast<float>(y) + 2 * z_re * z_im
//                 })
//             }))
//         }),
//         Vectorize(8),
//         Parallelize(8));
//     s = ProducerConsumer::make_produce(h.name(), s);    
// }

int main(int argc, char *argv[]) {
    // store_to_scalar();
    // load_store_scalar();
    // call_provide_scalar();
    // provide_loop();
    // provide_loop_multidim();
    // call_provide_loop_multidim();
    // in_place_bubble_sort();
    // add_vectorize();
    // add_parallelize_vectorize();
    test_break();
    // mandelbrot();
    return 0;
}
