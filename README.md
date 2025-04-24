# Parallel Matrix Multiplication using Hybrid MPI + OpenMP (via multiprocessing + threading in Python)

from mpi4py import MPI
import numpy as np
import threading
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix size (NxN)
N = 512

# Number of OpenMP-style threads per MPI process
NUM_THREADS = 4

# Generate matrices A and B
if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
else:
    A = None
    B = np.empty((N, N), dtype='d')

# Scatter rows of A to all processes
local_A = np.empty((N // size, N), dtype='d')
comm.Scatter(A, local_A, root=0)

# Broadcast matrix B to all processes
comm.Bcast(B, root=0)

# Placeholder for local result
local_C = np.zeros((N // size, N), dtype='d')

def compute_block(start_row, end_row):
    for i in range(start_row, end_row):
        for j in range(N):
            local_C[i][j] = np.dot(local_A[i], B[:, j])

# Threading for parallel computation within the process
threads = []
rows_per_thread = (N // size) // NUM_THREADS
start = time.time()
for t in range(NUM_THREADS):
    start_row = t * rows_per_thread
    end_row = (t + 1) * rows_per_thread if t != NUM_THREADS - 1 else N // size
    thread = threading.Thread(target=compute_block, args=(start_row, end_row))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
end = time.time()

# Gather local_C from all processes
if rank == 0:
    C = np.empty((N, N), dtype='d')
else:
    C = None

comm.Gather(local_C, C, root=0)

# Print timing and validate
if rank == 0:
    print(f"Parallel matrix multiplication completed in {end - start:.2f} seconds.")
    # Optionally validate with numpy's matmul
    C_seq = np.matmul(A, B)
    error = np.max(np.abs(C - C_seq))
    print(f"Maximum error compared to NumPy: {error:.6e}")
