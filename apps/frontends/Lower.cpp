#include "Lower.h"

using std::vector;

namespace Halide {
namespace Internal {

Module lower_from_stmt(Stmt s,
                       const std::string &pipeline_name,
                       const Target &t,
                       const std::vector<Argument> &args,
                       const LinkageType linkage_type,
                       const std::vector<Stmt> &requirements,
                       bool trace_pipeline,
                       const std::vector<IRMutator *> &custom_passes) {
    debug(1) << "Unpacking buffer arguments...\n";
    s = unpack_buffers(s);
    debug(2) << "Lowering after unpacking buffer arguments...\n"
             << s << "\n\n";

    std::vector<std::string> namespaces;
    std::string simple_pipeline_name = extract_namespaces(pipeline_name, namespaces);

    Module result_module(simple_pipeline_name, t);

    std::vector<Argument> public_args = args;
    LoweredFunc main_func(pipeline_name, public_args, s, linkage_type);

    // If we're in debug mode, add code that prints the args.
    if (t.has_feature(Target::Debug)) {
        debug_arguments(&main_func, t);
    }

    result_module.append(main_func);

    // Append a wrapper for this pipeline that accepts old buffer_ts
    // and upgrades them. It will use the same name, so it will
    // require C++ linkage. We don't need it when jitting.
    if (!t.has_feature(Target::JIT)) {
        add_legacy_wrapper(result_module, main_func);
    }

    return result_module;
}

} // namespace Internal
} // namespace Halide