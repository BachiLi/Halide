#ifndef _DISTRIBUTED_H
#define _DISTRIBUTED_H

#include <stdio.h>
#include "HalideBuffer.h"

extern int rank, numranks;
void distributed_init(int *argc, char ***argv);
void distributed_done(void);
Halide::Runtime::Buffer<uint8_t> *gather_to_rank0(Halide::Runtime::Buffer<uint8_t> &output,
    int full_image_width, int full_image_height);

// only print message on rank 0
#define fprintf_rank0(a, ...) do { if(!rank) fprintf(a, __VA_ARGS__); } while(0)
// print with "N: " prefix (where "N" is the node rank)
#define fprintf_rank(a, b, ...) fprintf(a, "%d: " b, rank, ##__VA_ARGS__)

#endif /* _DISTRIBUTED_H */
