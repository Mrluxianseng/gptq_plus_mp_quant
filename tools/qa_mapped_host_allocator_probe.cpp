// Small validation-only allocator. Runtime ABI declarations are checked against
// the installed CUDA 12.8 headers; no CUDA kernel or model math is implemented.
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <atomic>

extern "C" {
int cudaSetDevice(int);
int cudaSetDeviceFlags(unsigned int);
int cudaDeviceGetAttribute(int*, int, int);
int cudaHostAlloc(void**, size_t, unsigned int);
int cudaHostGetDevicePointer(void**, void*, unsigned int);
int cudaFreeHost(void*);
}

struct HostBlock { void* host; size_t bytes; };
static std::mutex block_mutex;
static std::unordered_map<void*,HostBlock> blocks;
static std::atomic<int> last_error{0};
static size_t live_bytes=0;
static size_t peak_bytes=0;
static size_t allocations=0;
static size_t releases=0;

extern "C" int qa_enable_map_host() {
    int result=cudaSetDeviceFlags(8u); // cudaDeviceMapHost, before deferred context creation.
    last_error.store(result); return result;
}
extern "C" int qa_device_attribute(int attribute, int device) {
    int value=-1; int result=cudaDeviceGetAttribute(&value,attribute,device);
    last_error.store(result); return result ? -1 : value;
}
extern "C" void* qa_mapped_alloc(size_t size, int device, void*) {
    int result=cudaSetDevice(device);
    void* host=nullptr;void* mapped=nullptr;
    if (!result) result=cudaHostAlloc(&host,size,3u); // Portable + Mapped.
    if (!result) result=cudaHostGetDevicePointer(&mapped,host,0u);
    if (result) {
        if (host) cudaFreeHost(host);
        last_error.store(result);return nullptr;
    }
    std::lock_guard<std::mutex> guard(block_mutex);
    blocks.emplace(mapped,HostBlock{host,size});
    live_bytes+=size;peak_bytes=live_bytes>peak_bytes?live_bytes:peak_bytes;
    ++allocations;last_error.store(0);return mapped;
}
// Four arguments: actual installed CUDAPluggableAllocator.h callback ABI.
extern "C" void qa_mapped_free(void* ptr, size_t, int, void*) {
    std::lock_guard<std::mutex> guard(block_mutex);
    auto found=blocks.find(ptr);
    if (found==blocks.end()) { last_error.store(-10);return; }
    int result=cudaFreeHost(found->second.host);
    last_error.store(result);
    if (!result) { live_bytes-=found->second.bytes;blocks.erase(found);++releases; }
}
extern "C" int qa_last_error() { return last_error.load(); }
extern "C" size_t qa_live_bytes() { std::lock_guard<std::mutex> guard(block_mutex);return live_bytes; }
extern "C" size_t qa_peak_bytes() { std::lock_guard<std::mutex> guard(block_mutex);return peak_bytes; }
extern "C" size_t qa_allocation_count() { std::lock_guard<std::mutex> guard(block_mutex);return allocations; }
extern "C" size_t qa_release_count() { std::lock_guard<std::mutex> guard(block_mutex);return releases; }
