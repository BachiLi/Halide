#include <mpi.h>
#include "distributed.h"

int rank = 0, numranks = 0;

std::vector<MPI_Request*> sends;
using namespace Halide::Runtime;

void distributed_init(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
}

void distributed_done(void) {
    // wait for outgoing transfers to complete
    for (MPI_Request *req : sends) {
        MPI_Wait(req, MPI_STATUS_IGNORE);
        delete req;
    }
    sends.clear();

    MPI_Finalize();
}

void report_mpi_error(MPI_Status *status, const char *when) {
    char str[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(status->MPI_ERROR, str, &len);
    fprintf(stderr, "%s, node %d, status %d (%s)\n", when, status->MPI_SOURCE, status->MPI_ERROR, str);
    MPI_Abort(MPI_COMM_WORLD, status->MPI_ERROR);
}

struct node_size {
    int xmin, xmax;
    int ymin, ymax;
    int cmin, cmax;
};

#define TAG_SIZE(node) (node)
#define TAG_STRIPE(node) (numranks+node)

Buffer<uint8_t> *gather_to_rank0(Buffer<uint8_t> &output, int full_image_width, int full_image_height) {
    if(numranks == 1) {
        // nothing to gather, use the local buffer as-is.
        return &output;
    }

    int local_xmin = output.dim(0).min();
    int local_xmax = output.dim(0).min()+output.dim(0).extent();
    int local_ymin = output.dim(1).min();
    int local_ymax = output.dim(1).min()+output.dim(1).extent();
    int local_cmin = output.dim(2).min();
    int local_cmax = output.dim(2).min()+output.dim(2).extent();
    int local_width  = local_xmax - local_xmin;
    int local_height = local_ymax - local_ymin;
    int local_colors = local_cmax - local_cmin;

    // each node sends its buffer size to node 0
    struct node_size my_sizes = {
        .xmin=local_xmin, .xmax=local_xmax,
        .ymin=local_ymin, .ymax=local_ymax,
        .cmin=local_cmin, .cmax=local_cmax
    };

    struct node_size node_sizes[numranks];
    MPI_Request node_size_reqs[numranks];
    MPI_Request *node_size_send = new MPI_Request;
    // fprintf_rank(stderr, "sending sizes to node 0\n");
    MPI_Isend(&my_sizes, sizeof(my_sizes), MPI_BYTE, 0, TAG_SIZE(rank), MPI_COMM_WORLD, node_size_send);
    sends.push_back(node_size_send);
    if(rank) {
        // fprintf_rank(stderr, "sending tile data to node 0\n");
        int len = local_width * local_height * local_colors;
        assert(len == output.number_of_elements());
        MPI_Send(output.begin(), len, MPI_BYTE, 0, TAG_STRIPE(rank), MPI_COMM_WORLD);
    }
    Buffer<uint8_t> *full_output = NULL;
    if(!rank) {
        // asynchronously receive shapes from nodes
        for(int node = 0; node < numranks; node++) {
            // fprintf_rank(stderr, "receiving sizes from node %d\n", node);
            MPI_Irecv(&node_sizes[node], sizeof(my_sizes), MPI_BYTE, node, TAG_SIZE(node), MPI_COMM_WORLD, &node_size_reqs[node]);
        }
        fprintf_rank(stderr, "gathering output data\n");
        full_output = new Buffer<uint8_t>(full_image_width, full_image_height, 3);
        for(int node = 0; node < numranks; node++) {
            MPI_Status status;
            memset(&status, 0, sizeof(status));
            MPI_Wait(&node_size_reqs[node], &status);
            if(status.MPI_ERROR) {
                report_mpi_error(&status, "MPI_Wait receiving sizing data");
            }

            struct node_size *node_size = &node_sizes[node];
            int node_width  = node_size->xmax - node_size->xmin;
            int node_height = node_size->ymax - node_size->ymin;
            int node_colors = node_size->cmax - node_size->cmin;
            halide_dimension_t shape[3];
            shape[0].min = node_size->xmin;
            shape[1].min = node_size->ymin;
            shape[2].min = node_size->cmin;
            shape[0].extent = node_size->xmax - node_size->xmin;
            shape[1].extent = node_size->ymax - node_size->ymin;
            shape[2].extent = node_size->cmax - node_size->cmin;
            int xmin = node_size->xmin;
            int xlen = node_size->xmax - node_size->xmin;
            Buffer<uint8_t> *source_buffer;
            Buffer<uint8_t> node_buffer(NULL, {node_width, node_height, node_colors});
            if(node == rank) {
                source_buffer = &output;
            } else {
                source_buffer = &node_buffer;
                node_buffer.allocate();
                uint8_t *ptr = node_buffer.begin();
                int buffer_len = node_buffer.number_of_elements();
                MPI_Status status;
                memset(&status, 0, sizeof(status));
                // fprintf_rank(stderr, "receiving tile position [%d,%d,%d] shape [%d,%d,%d] from node %d\n",
                //              node_size->xmin, node_size->ymin, node_size->cmin,
                //              node_width, node_height, node_colors, node);
                MPI_Recv(ptr, buffer_len, MPI_BYTE, node, TAG_STRIPE(node), MPI_COMM_WORLD, &status);
                if(status.MPI_ERROR) {
                    report_mpi_error(&status, "MPI_Recv receiving tile data");
                }
            }
            // copy tile data into the output buffer, one row at a time
            for(int c = 0; c < node_colors; c++) {
                for(int y = 0; y < node_height; y++) {
                    uint8_t *outptr = &(*full_output)(xmin, y+node_size->ymin, c+node_size->cmin);
                    uint8_t *tmpptr = &(*source_buffer)(0, y, c);
                    // fprintf_rank(stderr, "copying vector (x,%d,%d) of length %d from %p to %p\n",
                    //              y+node_size->ymin, c+node_size->cmin, xlen, tmpptr, outptr);
                    memcpy(outptr, tmpptr, xlen);
                }
            }
            node_buffer.deallocate();
        }
    }

    // completion of asynchronous sends is handled by distributed_done(), above.

    return full_output;
}
