#pragma once

#include "Halide.h"

namespace Halide {
namespace Internal {

// Doesn't do storage permutation yet.
Stmt storage_flattening(Stmt s,
						const std::map<std::string, Parameter> &output_buffers,
                        const Target &target);

} // namespace Internal
} // namespace Halide