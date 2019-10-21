#include "Halide.h"

using namespace Halide;
using namespace Halide::Internal;

int main(int argc, char *argv[]) {
    // Func f("f");
    // f() = 0;
    // Target target = get_jit_target_from_environment();
    // target.set_feature(Target::NoBoundsQuery, true);
    // target.set_feature(Target::NoAsserts, true);
    // f.compile_jit(target);

    Buffer<int> out = Buffer<int>::make_scalar("f");

    Parameter f(Int(32), true /*is_buffer*/, 0, out.name());
    Stmt s;
    s = Store::make(f.name(), 100, 0, f, const_true(), ModulusRemainder());
    s = ProducerConsumer::make_produce(f.name(), s);

    std::cerr << s << std::endl;

    Target target = get_jit_target_from_environment();
    target.set_feature(Target::NoBoundsQuery, true);
    target.set_feature(Target::NoAsserts, true);
    target.set_feature(Target::JIT, true);

    const std::string &name = "foo";
    Module module = lower_from_stmt(s,
                                    name,
                                    target,
                                    {Argument(out)},
                                    LinkageType::ExternalPlusMetadata);
    auto func = module.get_function_by_name(name);
    JITModule jit_module(module, func);

    const void *raw_args[64] = {nullptr};
    const halide_buffer_t *buf = out.raw_buffer();
    raw_args[0] = (const void*)buf;
    int exit = jit_module.argv_function()(raw_args);
    _halide_user_assert(exit == 0);
    std::cerr << "out:" << out() << std::endl;
    return 0;
}
