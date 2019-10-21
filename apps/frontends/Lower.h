#pragma once

#include "Halide.h"

namespace Halide {
namespace Internal {

Module lower_from_stmt(Stmt stmt,
                       const std::string &pipeline_name,
                       const Target &t,
                       const std::vector<Argument> &args,
                       const LinkageType linkage_type,
                       const std::vector<Stmt> &requirements = std::vector<Stmt>(),
                       bool trace_pipeline = false,
                       const std::vector<IRMutator *> &custom_passes = std::vector<IRMutator *>());

} // namespace Internal
} // namespace Halide