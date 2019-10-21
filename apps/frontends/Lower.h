#pragma once

#include "Halide.h"

namespace Halide {
namespace Internal {

/** A lowering pass that takes a Stmt instead of a Function.
 */
Module lower_from_stmt(Stmt stmt,
                       const std::string &pipeline_name,
                       const std::map<std::string, Parameter> &output_buffers,
                       const Target &t,
                       const std::vector<Argument> &args,
                       const LinkageType linkage_type,
                       const std::vector<Stmt> &requirements = std::vector<Stmt>(),
                       bool trace_pipeline = false,
                       const std::vector<IRMutator *> &custom_passes = std::vector<IRMutator *>());

} // namespace Internal
} // namespace Halide