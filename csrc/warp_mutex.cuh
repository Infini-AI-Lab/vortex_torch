#pragma once

#include <cuda/std/semaphore>

// Warp-level mutex using semaphore
class WarpMutexSemaphoreImpl {
 public:
  __device__ WarpMutexSemaphoreImpl() : semaphore_(1) {}
  __device__ ~WarpMutexSemaphoreImpl() {}

  __device__ void Lock(uint32_t lane_id) {
    if (lane_id == 0) { semaphore_.acquire(); }
    __syncwarp();
  }

  __device__ void Unlock(uint32_t lane_id) {
    __syncwarp();
    if (lane_id == 0) { semaphore_.release(); }
  }

  __device__ bool TryLock(uint32_t lane_id) {
    bool acquired = false;
    if (lane_id == 0) { acquired = semaphore_.try_acquire(); }
    acquired = __shfl_sync(0xFFFFFFFF, acquired ? 1 : 0, 0);
    __syncwarp();
    return acquired;
  }

 private:
  cuda::binary_semaphore<cuda::thread_scope_device> semaphore_;
};

// Initialize set mutexes
__global__ inline void InitCacheSetMutexWarp(uint32_t n_set, void* mutex) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_set) {
    new (reinterpret_cast<WarpMutexSemaphoreImpl*>(mutex) + idx) WarpMutexSemaphoreImpl;
  }
}
