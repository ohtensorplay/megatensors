// SPDX-License-Identifier: Apache-2.0

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <fcntl.h>
#include <cstring>
#ifdef _MSC_VER
#include <io.h>
#include <malloc.h>
#include <share.h>
#include <stdio.h>
#include <cstdint>
#include <mutex>
#include <unordered_set>
// Windows-compatible posix_memalign
static inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
    *memptr = _aligned_malloc(size, alignment);
    return (*memptr) ? 0 : errno;
}
// Windows-compatible pread
static inline int64_t pread(int fd, void *buf, size_t count, int64_t offset) {
    int64_t cur = _lseeki64(fd, 0, 1 /*SEEK_CUR*/);
    if (cur < 0) return -1;
    if (_lseeki64(fd, offset, 0 /*SEEK_SET*/) < 0) return -1;
    int rd = _read(fd, buf, (unsigned int)count);
    _lseeki64(fd, cur, 0 /*SEEK_SET*/);
    return rd;
}
// --- Windows equivalents for dlfcn.h ---
#include <windows.h>
#define RTLD_LAZY    0
#define RTLD_GLOBAL  0
#define RTLD_LOCAL   0
#ifndef RTLD_NODELETE
#define RTLD_NODELETE 0x1000
#endif

static std::mutex g_nodelete_handles_mutex;
static std::unordered_set<void*> g_nodelete_handles;

static inline bool is_windows_path_like(const char* filename) {
    if (!filename || !filename[0]) return false;
    return std::strchr(filename, '\\') != nullptr ||
           std::strchr(filename, '/') != nullptr ||
           (std::strlen(filename) > 1 && filename[1] == ':');
}

static inline void* dlopen(const char* filename, int mode) {
    if (!filename) return nullptr;
    DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if (is_windows_path_like(filename)) {
        flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
    }
    void* handle = reinterpret_cast<void*>(LoadLibraryExA(filename, nullptr, flags));
    if (handle && (mode & RTLD_NODELETE)) {
        std::lock_guard<std::mutex> lock(g_nodelete_handles_mutex);
        g_nodelete_handles.insert(handle);
    }
    return handle;
}
static inline void* dlsym(void* handle, const char* symbol) {
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
}
static inline int dlclose(void* handle) {
    {
        std::lock_guard<std::mutex> lock(g_nodelete_handles_mutex);
        if (g_nodelete_handles.find(handle) != g_nodelete_handles.end()) {
            return 0;
        }
    }
    return FreeLibrary(reinterpret_cast<HMODULE>(handle)) ? 0 : -1;
}

// --- Windows equivalents for mmap/munmap ---
#define PROT_READ   1
#define MAP_PRIVATE 2
#define MAP_FAILED  ((void*)-1)

static inline void* mmap(void* /*addr*/, size_t length, int /*prot*/, int /*flags*/, int fd, int64_t offset) {
    HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
    DWORD offsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD offsetLow  = static_cast<DWORD>(offset & 0xFFFFFFFF);
    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) return MAP_FAILED;
    void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, offsetHigh, offsetLow, length);
    CloseHandle(hMapping);  // view keeps the mapping alive
    return ptr ? ptr : MAP_FAILED;
}
static inline int munmap(void* addr, size_t /*length*/) {
    return UnmapViewOfFile(addr) ? 0 : -1;
}

// Map POSIX names to MSVC equivalents
#define open  _open
#define close _close
#define write _write
#define lseek _lseeki64
#define O_RDONLY _O_RDONLY
#define O_RDWR _O_RDWR
#define O_WRONLY _O_WRONLY
#define O_CREAT _O_CREAT
#define O_TRUNC _O_TRUNC
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sendfile.h>
#include <chrono>
#include <dlfcn.h>
#endif
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <limits>
#include <set>
#include <zstd.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/sha.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>
#include <openssl/x509v3.h>

#include "gpu_compat.h"
#include "ext.hpp"

#define ALIGN 4096

#ifdef _MSC_VER
void init_dstorage_bindings(pybind11::module_&);
#endif

bool debug_log = false;  // non-static: fix Windows build
static bool enable_gil_release = false;

static cpp_metrics_t mc = {.bounce_buffer_bytes = 0};

/* cpu_mode functions: for tests and debugs */

static CUfileError_t cpu_cuFileDriverOpen() { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverClose() { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverSetMaxDirectIOSize(size_t) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverSetMaxPinnedMemSize(size_t) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileBufRegister(const void *, size_t, int) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileBufDeregister(const void *) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileHandleRegister(CUfileHandle_t * in, CUfileDescr_t *) {
    *in = reinterpret_cast<CUfileHandle_t *>(malloc(sizeof(CUfileHandle_t)));
    if (*in != nullptr) {
        return CUfileError_t{.err = CU_FILE_SUCCESS};
    }
    return CUfileError_t{.err = CU_FILE_INTERNAL_ERROR};
}
static void cpu_cuFileHandleDeregister(CUfileHandle_t h) {
    free(reinterpret_cast<void *>(h));
}
static cudaError_t cpu_cudaMemcpy(void * dst, const void * src, size_t size, enum cudaMemcpyKind) {
    std::memcpy(dst, src, size);
    return cudaSuccess;
}
static cudaError_t cpu_cudaMemcpyAsync(void * dst, const void * src, size_t size, enum cudaMemcpyKind kind, cudaStream_t) {
    std::memcpy(dst, src, size);
    return cudaSuccess;
}
static cudaError_t cpu_cudaDeviceSynchronize() { return cudaSuccess; }
static cudaError_t cpu_cudaHostAlloc(void ** p, size_t length, unsigned int) {
    if (posix_memalign(p, ALIGN, length) < 0) {
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}
static cudaError_t cpu_cudaFreeHost(void * p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
    return cudaSuccess;
}
static cudaError_t cpu_cudaDeviceGetPCIBusId(char * in, int s, int) {
    if (s > 0)
        in[0] = 0;
    return cudaSuccess;
}
static cudaError_t cpu_cudaSetDevice(int) { return cudaSuccess; }
static int cpu_numa_run_on_node(int) {return 0; }

ext_funcs_t cpu_fns = ext_funcs_t {
    .cuFileDriverOpen = cpu_cuFileDriverOpen,
    .cuFileDriverClose = cpu_cuFileDriverClose,
    .cuFileDriverSetMaxDirectIOSize = cpu_cuFileDriverSetMaxDirectIOSize,
    .cuFileDriverSetMaxPinnedMemSize = cpu_cuFileDriverSetMaxPinnedMemSize,
    .cuFileBufRegister = cpu_cuFileBufRegister,
    .cuFileBufDeregister = cpu_cuFileBufDeregister,
    .cuFileHandleRegister = cpu_cuFileHandleRegister,
    .cuFileHandleDeregister = cpu_cuFileHandleDeregister,
    .cuFileRead = nullptr,
    .cudaMemcpy = cpu_cudaMemcpy,
    .cudaMemcpyAsync = cpu_cudaMemcpyAsync,
    .cudaDeviceSynchronize = cpu_cudaDeviceSynchronize,
    .cudaHostAlloc = cpu_cudaHostAlloc,
    .cudaFreeHost = cpu_cudaFreeHost,
    .cudaDeviceGetPCIBusId = cpu_cudaDeviceGetPCIBusId,
    .numa_run_on_node = cpu_numa_run_on_node,
    .cudaSetDevice = cpu_cudaSetDevice,
    .cudaImportExternalMemory = nullptr,
    .cudaExternalMemoryGetMappedBuffer = nullptr,
    .cudaDestroyExternalMemory = nullptr,
};
ext_funcs_t cuda_fns;

static bool gpu_found = false;
static bool is_hip_runtime = false;
static bool cufile_found = false;

static int cufile_ver = 0;

template <typename T> void mydlsym(T** h, void* lib, std::string const& name) {
    *h = reinterpret_cast<T*>(dlsym(lib, name.c_str()));
}

// Try to load one GPU runtime library (CUDA or HIP). Returns true and sets
// gpu_found/is_hip_runtime on success; leaves them unchanged on failure.
static bool load_gpu_lib(const std::string& lib_name, bool is_hip, bool init_log, int mode) {
    cudaError_t (*get_device_count)(int*) = nullptr;
    const char* sym_count = is_hip ? HIP_SYM_GET_DEVICE_COUNT : CUDA_SYM_GET_DEVICE_COUNT;

    void* handle = dlopen(lib_name.c_str(), mode);
    if (!handle) {
        if (init_log) fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", lib_name.c_str());
        return false;
    }

    mydlsym(&get_device_count, handle, sym_count);
    if (!get_device_count) {
        if (init_log) fprintf(stderr, "[DEBUG] No %s in %s, fallback!\n", sym_count, lib_name.c_str());
        dlclose(handle);
        return false;
    }

    int count = 0;
    if (get_device_count(&count) != cudaSuccess) count = 0;
    if (init_log) fprintf(stderr, "[DEBUG] %s: device count=%d\n", lib_name.c_str(), count);
    if (count == 0) {
        dlclose(handle);
        return false;
    }

    mydlsym(&cuda_fns.cudaMemcpy,             handle, is_hip ? HIP_SYM_MEMCPY                : CUDA_SYM_MEMCPY);
    mydlsym(&cuda_fns.cudaMemcpyAsync,        handle, is_hip ? HIP_SYM_MEMCPY_ASYNC          : CUDA_SYM_MEMCPY_ASYNC);
    mydlsym(&cuda_fns.cudaDeviceSynchronize,  handle, is_hip ? HIP_SYM_DEVICE_SYNCHRONIZE    : CUDA_SYM_DEVICE_SYNCHRONIZE);
    mydlsym(&cuda_fns.cudaHostAlloc,          handle, is_hip ? HIP_SYM_HOST_ALLOC            : CUDA_SYM_HOST_ALLOC);
    mydlsym(&cuda_fns.cudaFreeHost,           handle, is_hip ? HIP_SYM_FREE_HOST             : CUDA_SYM_FREE_HOST);
    mydlsym(&cuda_fns.cudaDeviceGetPCIBusId,  handle, is_hip ? HIP_SYM_DEVICE_GET_PCI_BUS_ID : CUDA_SYM_DEVICE_GET_PCI_BUS_ID);
    mydlsym(&cuda_fns.cudaDeviceMalloc,       handle, is_hip ? HIP_SYM_DEVICE_MALLOC         : CUDA_SYM_DEVICE_MALLOC);
    mydlsym(&cuda_fns.cudaDeviceFree,         handle, is_hip ? HIP_SYM_DEVICE_FREE           : CUDA_SYM_DEVICE_FREE);
    mydlsym(&cuda_fns.cudaDriverGetVersion,   handle, is_hip ? HIP_SYM_DRIVER_GET_VERSION    : CUDA_SYM_DRIVER_GET_VERSION);
    mydlsym(&cuda_fns.cudaDeviceGetAttribute, handle, is_hip ? HIP_SYM_DEVICE_GET_ATTRIBUTE  : CUDA_SYM_DEVICE_GET_ATTRIBUTE);
    mydlsym(&cuda_fns.cudaSetDevice,          handle, is_hip ? HIP_SYM_SET_DEVICE            : CUDA_SYM_SET_DEVICE);

    // External memory interop is CUDA-only (used by Windows DirectStorage path)
    if (!is_hip) {
        mydlsym(&cuda_fns.cudaImportExternalMemory, handle, "cudaImportExternalMemory");
        mydlsym(&cuda_fns.cudaExternalMemoryGetMappedBuffer, handle, "cudaExternalMemoryGetMappedBuffer");
        mydlsym(&cuda_fns.cudaDestroyExternalMemory, handle, "cudaDestroyExternalMemory");
    } else {
        cuda_fns.cudaImportExternalMemory = nullptr;
        cuda_fns.cudaExternalMemoryGetMappedBuffer = nullptr;
        cuda_fns.cudaDestroyExternalMemory = nullptr;
    }

    bool success = cuda_fns.cudaMemcpy && cuda_fns.cudaDeviceSynchronize;
    success = success && cuda_fns.cudaHostAlloc && cuda_fns.cudaFreeHost;
    success = success && cuda_fns.cudaDeviceGetPCIBusId && cuda_fns.cudaDeviceMalloc;
    success = success && cuda_fns.cudaDeviceFree && cuda_fns.cudaDriverGetVersion;
    success = success && cuda_fns.cudaDeviceGetAttribute && cuda_fns.cudaSetDevice;

    dlclose(handle);

    if (!success) {
        if (init_log) fprintf(stderr, "[DEBUG] %s missing required GPU functions. fallback\n", lib_name.c_str());
        return false;
    }

    if (init_log) fprintf(stderr, "[DEBUG] loaded: %s (hip=%d)\n", lib_name.c_str(), (int)is_hip);
    gpu_found = true;
    is_hip_runtime = is_hip;
    return true;
}

static void load_library_functions(const std::string& cudart_override = "") {
#ifdef _MSC_VER
    const char* numaLib = nullptr;  // NUMA not available on Windows
#else
    const char* numaLib = "libnuma.so.1";
#endif
    bool init_log = getenv(ENV_ENABLE_INIT_LOG);
    int mode = RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE;

    if (numaLib) {
        void* handle_numa = dlopen(numaLib, mode);
        if (handle_numa) {
            mydlsym(&cpu_fns.numa_run_on_node, handle_numa, "numa_run_on_node");
            if (cpu_fns.numa_run_on_node) {
                cuda_fns.numa_run_on_node = cpu_fns.numa_run_on_node;
                if (init_log) {
                    fprintf(stderr, "[DEBUG] loaded: %s\n", numaLib);
                }
            }
            dlclose(handle_numa);
        }
    }
    if (!cpu_fns.numa_run_on_node) {
        if (init_log && numaLib) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", numaLib);
        }
        cpu_fns.numa_run_on_node = cpu_numa_run_on_node;
        cuda_fns.numa_run_on_node = cpu_numa_run_on_node;
    }

    if (!cudart_override.empty()) {
        // Caller specified exact library — detect platform from name
        bool is_hip = cudart_override.find("hip") != std::string::npos;
        load_gpu_lib(cudart_override, is_hip, init_log, mode);
    } else {
        // Universal detection: try CUDA first, then ROCm
        if (!load_gpu_lib(CUDA_RUNTIME_LIB, false, init_log, mode)) {
            load_gpu_lib(HIP_RUNTIME_LIB, true, init_log, mode);
        }
    }

    if (!gpu_found) {
        cuda_fns.cudaMemcpy = cpu_cudaMemcpy;
        cuda_fns.cudaDeviceSynchronize = cpu_cudaDeviceSynchronize;
        cuda_fns.cudaHostAlloc = cpu_cudaHostAlloc;
        cuda_fns.cudaFreeHost = cpu_cudaFreeHost;
        cuda_fns.cudaDeviceGetPCIBusId = cpu_cudaDeviceGetPCIBusId;
        cuda_fns.cudaSetDevice = cpu_cudaSetDevice;
        cuda_fns.cudaImportExternalMemory = nullptr;
        cuda_fns.cudaExternalMemoryGetMappedBuffer = nullptr;
        cuda_fns.cudaDestroyExternalMemory = nullptr;
    }

#ifdef _MSC_VER
    const char* gdsLib = nullptr; // neither cuFile nor hipFile on Windows
#else
    const char* gdsLib = is_hip_runtime ? HIPFILE_LIB : CUFILE_LIB;
#endif
    cufile_found = false;
    if (gpu_found && gdsLib) {
        const bool is_hip = is_hip_runtime;
        void* handle_gds = dlopen(gdsLib, mode);
        if (handle_gds) {
            if (!is_hip) {
                // Only cuFile exposes a version query; hipFile does not.
                CUfileError_t (*cuFileGetVersion)(int *);
                mydlsym(&cuFileGetVersion, handle_gds, CUFILE_SYM_GET_VERSION);
                if (cuFileGetVersion) {
                    int version;
                    CUfileError_t err = cuFileGetVersion(&version);
                    if (err.err == CU_FILE_SUCCESS) {
                        cufile_ver = version;
                    }
                }
                if (cufile_ver == 0) {
                    fprintf(stderr, "[WARN] %s is loaded but its version is unknown", gdsLib);
                }
            }
            mydlsym(&cuda_fns.cuFileDriverOpen, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_OPEN : CUFILE_SYM_DRIVER_OPEN);
            mydlsym(&cuda_fns.cuFileDriverClose, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_CLOSE : CUFILE_SYM_DRIVER_CLOSE);
            mydlsym(&cuda_fns.cuFileDriverSetMaxDirectIOSize, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_SET_MAX_DIO_SIZE : CUFILE_SYM_DRIVER_SET_MAX_DIO_SIZE);
            mydlsym(&cuda_fns.cuFileDriverSetMaxPinnedMemSize, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_SET_MAX_PIN_SIZE : CUFILE_SYM_DRIVER_SET_MAX_PIN_SIZE);
            mydlsym(&cuda_fns.cuFileBufRegister, handle_gds, is_hip ? HIPFILE_SYM_BUF_REGISTER : CUFILE_SYM_BUF_REGISTER);
            mydlsym(&cuda_fns.cuFileBufDeregister, handle_gds, is_hip ? HIPFILE_SYM_BUF_DEREGISTER : CUFILE_SYM_BUF_DEREGISTER);
            mydlsym(&cuda_fns.cuFileHandleRegister, handle_gds, is_hip ? HIPFILE_SYM_HANDLE_REGISTER : CUFILE_SYM_HANDLE_REGISTER);
            mydlsym(&cuda_fns.cuFileHandleDeregister, handle_gds, is_hip ? HIPFILE_SYM_HANDLE_DEREGISTER : CUFILE_SYM_HANDLE_DEREGISTER);
            mydlsym(&cuda_fns.cuFileRead, handle_gds, is_hip ? HIPFILE_SYM_READ : CUFILE_SYM_READ);
            bool success = cuda_fns.cuFileDriverOpen && cuda_fns.cuFileDriverClose && cuda_fns.cuFileDriverSetMaxDirectIOSize;
            success &= cuda_fns.cuFileDriverSetMaxPinnedMemSize && cuda_fns.cuFileBufRegister && cuda_fns.cuFileBufDeregister;
            success &= cuda_fns.cuFileHandleRegister && cuda_fns.cuFileHandleDeregister && cuda_fns.cuFileRead;
            if (!success) {
                if (init_log) {
                    fprintf(stderr, "[DEBUG] %s does not contain required GDS functions. fallback\n", gdsLib);
                }
            } else {
                if (init_log) {
                    if (is_hip) {
                        // hipFile has no version query (see above).
                        fprintf(stderr, "[DEBUG] loaded: %s\n", gdsLib);
                    } else {
                        fprintf(stderr, "[DEBUG] loaded: %s (ver: %d.%d.%d)\n", gdsLib, cufile_ver / 1000, (cufile_ver % 1000) / 10, cufile_ver % 10);
                    }
                }
                cufile_found = true;
            }
            dlclose(handle_gds);
        } else if (init_log) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", gdsLib);
        }
    }

    if (!cufile_found) {
        cuda_fns.cuFileDriverOpen = cpu_cuFileDriverOpen;
        cuda_fns.cuFileDriverClose = cpu_cuFileDriverClose;
        cuda_fns.cuFileDriverSetMaxDirectIOSize = cpu_cuFileDriverSetMaxDirectIOSize;
        cuda_fns.cuFileDriverSetMaxPinnedMemSize = cpu_cuFileDriverSetMaxPinnedMemSize;
        cuda_fns.cuFileBufRegister = cpu_cuFileBufRegister;
        cuda_fns.cuFileBufDeregister = cpu_cuFileBufDeregister;
        cuda_fns.cuFileHandleRegister = cpu_cuFileHandleRegister;
        cuda_fns.cuFileHandleDeregister = cpu_cuFileHandleDeregister;

        cuda_fns.cuFileRead = nullptr;
    }
}

bool is_cuda_found()
{
    return gpu_found && !is_hip_runtime;
}

bool is_hip_found()
{
    return gpu_found && is_hip_runtime;
}

bool is_cufile_found()
{
    return cufile_found;
}

/* The version is returned as (1000 * major + 10 * minor). */
int cufile_version()
{
    return cufile_ver;
}

int get_alignment_size()
{
    return ALIGN;
}

void set_debug_log(bool _debug_log)
{
    debug_log = _debug_log;
}

void set_gil_release(bool enable) {
    enable_gil_release = enable;
}

bool get_gil_release() {
    return enable_gil_release;
}

void init_gil_release_from_env() {
    const char* env_val = std::getenv("MEGATENSORS_ENABLE_GIL_RELEASE");
    if (env_val != nullptr) {
        std::string env_str(env_val);
        // Convert to lowercase for case-insensitive comparison
        std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
        enable_gil_release = (env_str == "1" || env_str == "true" || env_str == "yes" || env_str == "on");
        if (debug_log) {
            std::printf("[DEBUG] GIL release %s via environment variable MEGATENSORS_ENABLE_GIL_RELEASE=%s\n",
                       enable_gil_release ? "enabled" : "disabled", env_val);
        }
    }
}

int is_gds_supported(int deviceId)
{
    int gdr_support = 1;
    int driverVersion = 0;

    cudaError_t err = cuda_fns.cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "is_gds_supported: %s failed, deviceId=%d, err=%d\n",
            is_hip_runtime ? HIP_SYM_DRIVER_GET_VERSION : CUDA_SYM_DRIVER_GET_VERSION, deviceId, err);
        return -1;
    }

    if (is_hip_runtime) {
        // hipFile requires ROCm >= 7.2.
        constexpr int HIPFILE_MIN_HIP_VER = 70200000;
        if (!cufile_found || driverVersion < HIPFILE_MIN_HIP_VER) return 0;
        return gdr_support;
    }

    if (driverVersion > 11030) {
        err = cuda_fns.cudaDeviceGetAttribute(&gdr_support, cudaDevAttrGPUDirectRDMASupported, deviceId);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "is_gds_supported: cudaDeviceGetAttribute failed, deviceId=%d, err=%d\n", deviceId, err);
            return -1;
        }
    }
    return gdr_support;
}

int init_gds()
{
    CUfileError_t err;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (cuda_fns.cuFileDriverOpen) {
        err = cuda_fns.cuFileDriverOpen();
        if (err.err != CU_FILE_SUCCESS) {
            std::fprintf(stderr, "init_gds: cuFileDriverOpen returned an error = %d\n", err.err);
            return -1;
        }
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] init_gds: cuFileDriverOpen=%" PRId64 " us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

int close_gds()
{
    CUfileError_t err;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (cuda_fns.cuFileDriverClose) {
        err = cuda_fns.cuFileDriverClose();
        if (err.err != CU_FILE_SUCCESS) {
            std::fprintf(stderr, "close_gds: cuFileDriverClose returned an error = %d\n", err.err);
            return -1;
        }
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] close_gds: cuFileDriverClose, elapsed=%" PRId64 " us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

std::string get_device_pci_bus(int deviceId) {
    cudaError_t err;
    char pciBusId[32];

    std::memset(pciBusId, 0, 32);
    if (cuda_fns.cudaDeviceGetPCIBusId) {
        err = cuda_fns.cudaDeviceGetPCIBusId(pciBusId, 32, deviceId);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "get_device_pci_bus: cudaDeviceGetPCIBusId failed, deviceId=%d, err=%d\n", deviceId, err);
            return "";
        }
    } else {
        return "";
    }
    return std::string(pciBusId);
}

int set_numa_node(int numa_node) {
    if (numa_node >= 0) {
        if (cpu_fns.numa_run_on_node(numa_node) != 0) {
            std::fprintf(stderr, "set_numa_node: numa_run_on_node(numa_node=%d) failed\n", numa_node);
            return -1;
        }
    }
    return 0;
}

pybind11::bytes read_buffer(uintptr_t _dst, uint64_t length) {
    std::string buf;
    char *c = reinterpret_cast<char *>(_dst);
    buf.insert(buf.end(), c, c+length);
    return pybind11::bytes(buf);
}

uintptr_t cpu_malloc(uint64_t length) {
    void *p;
    if (posix_memalign(&p, ALIGN, length) < 0) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(p);
}

void cpu_free(uintptr_t addr) {
    void *p = reinterpret_cast<void *>(addr);
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}

uintptr_t gpu_malloc(uint64_t length) {
    void *p;
    if (cuda_fns.cudaDeviceMalloc(&p, length) != cudaSuccess) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(p);
}

void gpu_free(uintptr_t addr) {
    cuda_fns.cudaDeviceFree(reinterpret_cast<void*>(addr));
}

const int gds_device_buffer::cufile_register(uint64_t offset, uint64_t length) {
    CUfileError_t err;
    void * dst = reinterpret_cast<void*>(this->_devPtr_base->get_uintptr() + offset);

    std::chrono::steady_clock::time_point begin_register = std::chrono::steady_clock::now();
    err = _fns->cuFileBufRegister(dst, length, 0);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "gds_device_buffer.cufile_register: cuFileBufRegister returned an error = %d\n", err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.cufile_register: addr=%p, offset=%" PRIu64 ", length=%" PRIu64 ", register=%" PRId64 " us\n", dst, offset, length,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin_register).count());
    }
    return 0;
}

const int gds_device_buffer::cufile_deregister(uint64_t offset) {
    void * dst = reinterpret_cast<void*>(this->_devPtr_base->get_uintptr() + offset);
    CUfileError_t err;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = _fns->cuFileBufDeregister(dst);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "gds_device_buffer.cufile_deregister: cuFileBufDeregister (%p) returned an error=%d\n", dst, err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.cufile_deregister: addr=%p, offset=%" PRIu64 ", elapsed=%" PRId64 " us\n", dst, offset,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

const int gds_device_buffer::memmove(uint64_t _dst_off, uint64_t _src_off, const gds_device_buffer& _tmp, uint64_t length) {
    cudaError_t err;
    void *dst = reinterpret_cast<void *>(this->_devPtr_base->get_uintptr() + _dst_off);
    void *src = reinterpret_cast<void *>(this->_devPtr_base->get_uintptr() + _src_off);
    void *tmp = const_cast<void *>(_tmp._devPtr_base->get_raw());

    if (this->_length < _dst_off) {
        std::fprintf(stderr, "gds_device_buffer.memmove: length is smaller than request dst_off, tmp.length=%" PRIu64 ", _dst_off=%" PRIu64 "\n", _tmp._length, _dst_off);
        return -1;
    }
    if (this->_length < _src_off) {
        std::fprintf(stderr, "gds_device_buffer.memmove: length is smaller than request dst_off, tmp.length=%" PRIu64 ", _src_off=%" PRIu64 "\n", _tmp._length, _src_off);
        return -1;
    }
    if (_tmp._length < length) {
        std::fprintf(stderr, "gds_device_buffer.memmove: tmp is smaller than request length, tmp.length=%" PRIu64 ", length=%" PRIu64 "\n", _tmp._length, length);
        return -1;
    }
    if (length == 0) {
        return 0;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = _fns->cudaMemcpy(tmp, src, length, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::printf("gds_device_buffer.memmove: cudaMemcpy[0](tmp=%p, src=%p, length=%" PRIu64 ") failed, err=%d\n", tmp, src, length, err);
        return -1;
    }
    err = _fns->cudaMemcpy(dst, tmp, length, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::printf("gds_device_buffer.memmove: cudaMemcpy[1](dst=%p, tmp=%p, length=%" PRIu64 ") failed, err=%d\n", dst, tmp, length, err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.memmove: dst=%p, src=%p, tmp=%p, length=%" PRIu64 ", elapsed=%" PRId64 " us\n", dst, src, tmp, length,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}


void nogds_file_reader::_thread(const int thread_id, ext_funcs_t *fns, const int device_id, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off, thread_states_t *s) {
    void * src = nullptr;
    void * mmap_base = nullptr;
    size_t mmap_length = 0;
    cudaError_t err;

    // Set the CUDA device for this thread. New std::threads do not inherit the
    // parent thread's CUDA device and default to device 0, which would create
    // an unwanted CUDA context on device 0.
    if (device_id >= 0) {
        fns->cudaSetDevice(device_id);
    }
    int64_t count;
    bool failed = false;
    void * buffer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(s->_read_buffer) + s->_bbuf_size_kb * 1024 * (thread_id % s->_max_threads));

    if (s->_use_mmap) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (offset < 0 || length < 0) {
            failed = true;
            goto out;
        }
#ifdef _MSC_VER
        const int64_t page_size = 65536;
#else
        const long sys_page_size = sysconf(_SC_PAGE_SIZE);
        const int64_t page_size = sys_page_size > 0 ? static_cast<int64_t>(sys_page_size) : 4096;
#endif
        const int64_t aligned_offset = (offset / page_size) * page_size;
        const int64_t offset_delta = offset - aligned_offset;
        mmap_length = static_cast<size_t>(length + offset_delta);
        mmap_base = mmap(NULL, mmap_length, PROT_READ, MAP_PRIVATE, fd, aligned_offset);
        if (mmap_base == MAP_FAILED) {
            std::printf("nogds_file_reader._thread: mmap(fd=%d, offset=%" PRIu64 ", length=%" PRIu64 ") failed\n", fd, offset, length);
            mmap_base = nullptr;
            failed = true;
            goto out;
        }
        src = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mmap_base) + static_cast<uintptr_t>(offset_delta));
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader._thread: mmap, fd=%d, offset=%" PRIu64 ", length=%" PRIu64 ", elapsed=%" PRId64 " us\n",
                fd, offset, length, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        }
    }
    count = 0;
    while (count < length) {
        int64_t l = length - count;
        int64_t c;
        if (l > (int64_t)(s->_bbuf_size_kb * 1024)) {
            l = (int64_t)(s->_bbuf_size_kb * 1024);
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (s->_use_mmap) {
            std::memcpy(buffer, (void *)((uintptr_t)src + count), l);
            c = l;
        } else {
            c = pread(fd, buffer, l, offset + count);
            if (c != l) {
                std::printf("nogds_file_reader._thread failed: pread(fd=%d, buffer=%p, offset=%" PRIu64 ", count=%" PRIi64 ", l=%" PRIi64 "), c=%" PRIi64 "\n", fd, buffer, offset, count, l, c);
                failed = true;
                goto out;
            }
        }
        std::chrono::steady_clock::time_point memcpy_begin = std::chrono::steady_clock::now();
        err = fns->cudaMemcpy(dst._get_raw_pointer(ptr_off + count, c), buffer, c, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::printf("nogds_file_reader._thread: cudaMemcpy(%p, %p, %" PRIi64 ") failed, err=%d\n", dst._get_raw_pointer(ptr_off + count, c), buffer, count, err);
            failed = true;
            goto out;
        } else if (c <= 64 * 1024) {
            fns->cudaDeviceSynchronize();
        }
        count += c;
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader._thread: read (mmap=%d), fd=%d, offset=%" PRIu64 ", count=%" PRIi64 ", c=%" PRIi64 ", copy=%" PRId64 " us, cuda_copy=%" PRId64 " us\n",
                s->_use_mmap, fd, offset, count, c, std::chrono::duration_cast<std::chrono::microseconds>(memcpy_begin - begin).count(), std::chrono::duration_cast<std::chrono::microseconds>(end - memcpy_begin).count());
        }
    }
out:
    {
        std::unique_lock lk(s->_result_mutex);
        if (failed) {
            s->_results[thread_id] = nullptr;
        } else {
            s->_results[thread_id] = dst._get_raw_pointer(ptr_off, length);
        }
        s->_result_cond.notify_one();
    }
    if (s->_use_mmap && mmap_base != nullptr) {
        munmap(mmap_base, mmap_length);
    }
}

const int nogds_file_reader::submit_read(const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off)
{
    const int thread_id = this->_next_thread_id++;
    if (this->_threads == nullptr) {
        this->_threads = new std::thread*[this->_s._max_threads];
        for (uint64_t i = 0; i < this->_s._max_threads; ++i) {
            this->_threads[i] = nullptr;
        }
    }
    if (this->_s._read_buffer == nullptr) {
        cudaError_t err;
        std::chrono::steady_clock::time_point alloc_begin = std::chrono::steady_clock::now();
        auto buf_len = this->_s._bbuf_size_kb * 1024 * this->_s._max_threads;
        err = _fns->cudaHostAlloc(&this->_s._read_buffer, buf_len, 0);
        if (err != cudaSuccess) {
            std::printf("nogds_file_reader.submit_read: cudaHostAlloc(%" PRIi64 ") failed\n", buf_len);
            return -1;
        }
        mc.bounce_buffer_bytes += buf_len;
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, addr=%p, size=%" PRIi64 ", elapsed=%" PRId64 " us\n",
                reinterpret_cast<void*>(this->_s._read_buffer),
                buf_len, std::chrono::duration_cast<std::chrono::microseconds>(end - alloc_begin).count());
        }
    }
    std::thread *t = this->_threads[thread_id % this->_s._max_threads];
    if (t != nullptr) {
        t->join();
        delete(t);
    }
    t = new std::thread(nogds_file_reader::_thread, thread_id, _fns, this->_device_id, fd, dst, offset, length, ptr_off, &this->_s);
    this->_threads[thread_id % this->_s._max_threads] = t;
    if (debug_log) {
        std::printf("[DEBUG] nogds_file_reader.submit_read #3, thread_id=%d\n", thread_id);
    }
    return thread_id;
}

const uintptr_t nogds_file_reader::wait_read(const int thread_id) {
    void * ret;
    {
        std::unique_lock lk(this->_s._result_mutex);
        while(this->_s._results.count(thread_id) == 0) {
            this->_s._result_cond.wait(lk);
        }
        ret = this->_s._results.at(thread_id);
        this->_s._results.erase(thread_id);
    }
    return reinterpret_cast<const uintptr_t>(ret);
}

nogds_file_reader::~nogds_file_reader() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->_s._read_buffer != nullptr) {
        auto buf_len = this->_s._bbuf_size_kb * 1024 * this->_s._max_threads;
        _fns->cudaFreeHost(this->_s._read_buffer);
        if (debug_log) {
            std::printf("[DEBUG] cudaFreeHost, addr=%p, size=%" PRIi64 "\n",
                reinterpret_cast<void *>(this->_s._read_buffer), buf_len);
        }
        this->_s._read_buffer = nullptr;
        mc.bounce_buffer_bytes -= buf_len;
    }
    if (this->_threads != nullptr) {
        for (uint64_t i = 0; i < this->_s._max_threads; ++i) {
            std::thread * t = this->_threads[i];
            if (t != nullptr) {
                t->join();
                delete(t);
            }
        }
        delete(this->_threads);
        this->_threads = nullptr;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] ~nogds_file_reader: elapsed=%" PRId64 " us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
}

raw_gds_file_handle::raw_gds_file_handle(std::string filename, bool o_direct, bool use_cuda) {
    CUfileHandle_t cf_handle;
    CUfileDescr_t cf_descr;
    CUfileError_t err;
    int fd;
    int flags = O_RDONLY;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#if defined(O_DIRECT)
    if (o_direct) {
        flags |= O_DIRECT;
    }
#endif
    fd = open(filename.c_str(), flags, 0644);
    if (fd < 0) {
        char msg[256];
        std::snprintf(msg, 256, "raw_gds_file_handle: open returned an error = %d", errno);
        throw std::runtime_error(msg);
    }
    std::memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    _fns = use_cuda ? &cuda_fns: &cpu_fns;

    err = _fns->cuFileHandleRegister(&cf_handle, &cf_descr);
    if (err.err != CU_FILE_SUCCESS) {
        close(fd);
        char msg[256];
        std::snprintf(msg, 256, "raw_gds_file_handle: cuFileHandleRegister returned an error = %d", err.err);
        throw std::runtime_error(msg);
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] raw_gds_file_handle: fd=%d, cf_handle=%p, elapsed=%" PRId64 " us\n", fd, cf_handle,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    this->_cf_handle = cf_handle;
    this->_fd = fd;
}

raw_gds_file_handle::~raw_gds_file_handle() {
    if (this->_cf_handle != 0) {
        _fns->cuFileHandleDeregister(this->_cf_handle);
        if (debug_log) {
            std::printf("[DEBUG] ~raw_gds_file_handle: cuFileHandleDeregister: cf_handle=%p\n", this->_cf_handle);
        }
    }
    if (this->_fd > 0) {
        close(this->_fd);
        if (debug_log) {
            std::printf("[DEBUG] ~raw_gds_file_handle: close: fd=%d\n", this->_fd);
        }
    }
}

void gds_file_reader::_thread(const int thread_id, ext_funcs_t *fns, const int device_id, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length, thread_states_t *s) {
    // Set the CUDA device for this thread. New std::threads do not inherit the
    // parent thread's CUDA device and default to device 0, which would create
    // an unwanted CUDA context on device 0.
    if (device_id >= 0) {
        fns->cudaSetDevice(device_id);
    }
    ssize_t count = 0;
    void * devPtr_base = dst._get_raw_pointer(ptr_off, length);
    std::chrono::steady_clock::time_point begin, begin_notify;

    // NOTE: we cannot call register_buffer here since it apparently fails when cuFileRead runs in background.
    begin = std::chrono::steady_clock::now();
    while (uint64_t(count) < length && offset + uint64_t(count) < file_length) {
        ssize_t c;
        if (!fns->cuFileRead) {
            c = pread(fh._get_fd(), reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(devPtr_base) + count), length - count, offset + count);
        } else {
            c = fns->cuFileRead(fh._get_cf_handle(), devPtr_base, length - count, offset + count, count);
        }
        if (debug_log) {
            std::printf("[DEBUG] gds_file_reader._thread: cuFileRead(fh, %p, length=%" PRIu64 ", off=%" PRIu64 ", ptr_off=%" PRIu64 ", count=%zd)=%zd\n", devPtr_base, length, offset, ptr_off, count, c);
        }
        if (c < 0) {
            std::fprintf(stderr, "gds_file_reader._thread: cuFileRead returned an error: errno=%d\n", errno);
            count = -1;
            break;
        } else if (c == 0) {
            break;
        }
        count += size_t(c);
    }
    begin_notify = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> guard(s->_result_lock);
        s->_results.insert(std::make_pair(thread_id, count));
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_file_reader._thread: fh=%p, offset=%" PRIu64 ", length=%" PRIu64 ", count=%zd, read=%" PRId64" us, notify=%" PRId64 " us\n",
            fh._get_cf_handle(), offset, length, count,
            std::chrono::duration_cast<std::chrono::microseconds>(begin_notify - begin).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin_notify).count());
    }
}

const int gds_file_reader::submit_read(const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length) {
    int id;
    std::thread * t;

    id = this->_next_id++;
    size_t thread_index = (size_t)(id % this->_s._max_threads);

    if (this->_threads == nullptr) {
        this->_threads = new std::thread*[this->_s._max_threads];
        for (int i = 0; i < this->_s._max_threads; i++) {
            this->_threads[i] = nullptr;
        }
    }

    t = this->_threads[thread_index];
    if (t != nullptr) {
        // block if we have too many readers
        // NOTE: caller (i.e., python code) runs on a single thread.  so, we do not care about more than two waiters
        t->join();
        delete(t);
    }
    t = new std::thread(_thread, id, _fns, this->_device_id, fh, dst, offset, length, ptr_off, file_length, &this->_s);
    this->_threads[thread_index] = t;
    return id;
}

const ssize_t gds_file_reader::wait_read(const int id) {
    size_t thread_index = (size_t)(id % this->_s._max_threads);
    if (this->_threads != nullptr) {
        std::thread * t = this->_threads[thread_index];
        if (t != nullptr) {
            t->join();
            delete(t);
            this->_threads[thread_index] = nullptr;
        }
    }
    std::lock_guard<std::mutex> guard(this->_s._result_lock);
    ssize_t ret = this->_s._results.at(id);
    this->_s._results.erase(id);
    return ret;
}

cpp_metrics_t get_cpp_metrics() {
    return mc;
}

namespace {

constexpr uint32_t MEGA_MAGIC = 0x4147454dU;
constexpr uint32_t MEGAKV_MAGIC = 0x564b474dU;
constexpr uint32_t MEGA_FOOTER_MAGIC = 0x4654474dU;
constexpr uint64_t MEGA_FOOTER_TRAILER_SIZE = 68;
constexpr uint64_t MEGA_DEFAULT_ALIGNMENT = 32;

enum mega_metadata_value_type : uint32_t {
    MEGA_META_UINT8 = 0,
    MEGA_META_INT8 = 1,
    MEGA_META_UINT16 = 2,
    MEGA_META_INT16 = 3,
    MEGA_META_UINT32 = 4,
    MEGA_META_INT32 = 5,
    MEGA_META_UINT64 = 6,
    MEGA_META_INT64 = 7,
    MEGA_META_FLOAT32 = 8,
    MEGA_META_FLOAT64 = 9,
    MEGA_META_BOOL = 10,
    MEGA_META_STRING = 11,
    MEGA_META_ARRAY = 12,
};

constexpr uint32_t MEGA_TENSOR_FLAG_COMPRESSED = 1;
constexpr uint32_t MEGA_TENSOR_FLAG_BYTE_SHUFFLED = 2;
constexpr uint32_t MEGA_COMPRESSION_NONE = 0;
constexpr uint32_t MEGA_COMPRESSION_ZSTD = 1;
constexpr uint32_t MEGA_COMPRESSION_AUTO = 0xFFFFFFFFU;
constexpr uint32_t MEGA_CHECKSUM_NONE = 0;
constexpr uint32_t MEGA_CHECKSUM_CRC32 = 1;
constexpr uint32_t MEGA_CHECKSUM_SHA256 = 2;
constexpr uint32_t MEGA_NO_SEGMENT_ID = UINT32_MAX;
constexpr uint32_t MEGA_FORMAT_VERSION = 1;
constexpr uint32_t MEGA_TENSOR_DIR_HAS_STORAGE_FORMAT = 1u << 0;
constexpr uint32_t MEGA_TENSOR_DIR_HAS_STORED_NBYTES = 1u << 1;
constexpr uint32_t MEGA_TENSOR_DIR_HAS_TENSOR_FLAGS = 1u << 2;
constexpr uint32_t MEGA_TENSOR_DIR_HAS_COMPRESSION_CODEC = 1u << 3;
constexpr uint32_t MEGA_TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE = 1u << 4;
constexpr uint32_t MEGA_TENSOR_DIR_HAS_CHECKSUM = 1u << 5;

struct mega_chunk_record {
    uint32_t tensor_id;
    uint32_t chunk_id;
    uint64_t logical_offset;
    uint64_t logical_size;
    uint64_t payload_offset;
    uint64_t stored_size;
    uint32_t codec;
    uint32_t flags;
    uint32_t checksum_type;
    std::array<uint8_t, 32> checksum;
};

struct mega_segment_record {
    uint32_t segment_id;
    uint32_t kind;
    uint32_t priority;
    uint32_t flags;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t expert_start;
    uint32_t expert_end;
    uint64_t payload_offset;
    uint64_t payload_size;
    uint32_t first_tensor_id;
    uint32_t tensor_count;
    uint32_t prefetch_group;
    uint32_t device_hint;
    uint32_t cache_policy;
};

struct mega_moe_expert_record {
    uint32_t tensor_id;
    uint32_t layer_id;
    uint32_t expert_id;
    uint32_t expert_role;
    uint32_t segment_id;
    uint32_t prefetch_group;
    uint32_t cache_policy;
};

struct megakv_entry_record {
    std::string name;
    uint32_t layer_id;
    uint32_t kv_role;
    uint64_t sequence_id;
    uint64_t token_start;
    uint64_t token_count;
    std::string logical_dtype;
    std::vector<uint64_t> shape;
    uint64_t payload_offset;
    uint64_t logical_nbytes;
    uint64_t stored_nbytes;
    uint32_t flags;
    uint32_t codec;
    uint32_t shuffle_elem_size;
    uint32_t checksum_type;
    std::array<uint8_t, 32> checksum;
};

struct mega_footer_overlay {
    bool present = false;
    uint64_t generation = 0;
    uint64_t overlay_offset = 0;
    uint64_t overlay_size = 0;
    uint64_t base_size = 0;
    std::array<uint8_t, 32> checksum{};
    pybind11::dict metadata;
};

static void read_exact_at(int fd, uint64_t offset, uint8_t *dst, uint64_t length, const std::string &filename) {
    uint64_t done = 0;
    while (done < length) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(length - done, 1ULL << 30));
        const int64_t n = pread(fd, dst + done, chunk, offset + done);
        if (n <= 0) {
            throw std::runtime_error(filename + ": failed to read MEGA tensor payload");
        }
        done += static_cast<uint64_t>(n);
    }
}

static std::vector<uint8_t> decompress_zstd(const std::vector<uint8_t> &payload, uint64_t logical_nbytes) {
    std::vector<uint8_t> decoded(static_cast<size_t>(logical_nbytes));
    const size_t rc = ZSTD_decompress(decoded.data(), decoded.size(), payload.data(), payload.size());
    if (ZSTD_isError(rc) || rc != static_cast<size_t>(logical_nbytes)) {
        const char *err = ZSTD_isError(rc) ? ZSTD_getErrorName(rc) : "decoded length mismatch";
        throw std::runtime_error(std::string("MEGA ZSTD payload decompression failed: ") + err);
    }
    return decoded;
}

static std::vector<uint8_t> decompress_payload(uint32_t codec, const std::vector<uint8_t> &payload, uint64_t logical_nbytes) {
    switch (codec) {
        case MEGA_COMPRESSION_NONE:
            return payload;
        case MEGA_COMPRESSION_ZSTD:
            return decompress_zstd(payload, logical_nbytes);
        default:
            throw std::runtime_error("MEGA payload uses unknown compression codec; only ZSTD is supported");
    }
}

static std::vector<uint8_t> compress_zstd_payload(const std::vector<uint8_t> &payload, int level) {
    const size_t bound = ZSTD_compressBound(payload.size());
    std::vector<uint8_t> out(bound);
    const size_t rc = ZSTD_compress(out.data(), out.size(), payload.data(), payload.size(), level);
    if (ZSTD_isError(rc)) {
        throw std::runtime_error(std::string("MEGA ZSTD payload compression failed: ") + ZSTD_getErrorName(rc));
    }
    out.resize(rc);
    return out;
}

static std::vector<uint8_t> compress_payload(uint32_t codec, const std::vector<uint8_t> &payload, int level) {
    switch (codec) {
        case MEGA_COMPRESSION_ZSTD:
            return compress_zstd_payload(payload, level);
        default:
            throw std::runtime_error("MEGA payload uses unknown compression codec; only ZSTD is supported");
    }
}

static bool checksum_tail_is_zero(const std::array<uint8_t, 32> &checksum, size_t first_checked_byte) {
    for (size_t i = first_checked_byte; i < checksum.size(); ++i) {
        if (checksum[i] != 0) {
            return false;
        }
    }
    return true;
}

static uint32_t read_checksum_u32_le(const std::array<uint8_t, 32> &checksum) {
    return uint32_t(checksum[0]) |
           (uint32_t(checksum[1]) << 8) |
           (uint32_t(checksum[2]) << 16) |
           (uint32_t(checksum[3]) << 24);
}

static std::string openssl_error_string();

static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t size) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> out{};
        for (uint32_t i = 0; i < out.size(); ++i) {
            uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit) {
                c = (c & 1U) ? (0xEDB88320U ^ (c >> 1U)) : (c >> 1U);
            }
            out[i] = c;
        }
        return out;
    }();
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8U);
    }
    return ~crc;
}

static uint32_t crc32_bytes(const std::vector<uint8_t> &payload) {
    return crc32_update(0, payload.data(), payload.size());
}

static void append_bytes(std::vector<uint8_t> &dst, const void *src, size_t len) {
    const uint8_t *begin = reinterpret_cast<const uint8_t *>(src);
    dst.insert(dst.end(), begin, begin + len);
}

template <typename T>
static void append_le(std::vector<uint8_t> &dst, T value) {
    append_bytes(dst, &value, sizeof(T));
}

static void append_string(std::vector<uint8_t> &dst, const std::string &value) {
    append_le<uint64_t>(dst, static_cast<uint64_t>(value.size()));
    append_bytes(dst, value.data(), value.size());
}

static void append_compact_string(std::vector<uint8_t> &dst, const std::string &value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("MEGA compact string too large");
    }
    append_le<uint32_t>(dst, static_cast<uint32_t>(value.size()));
    append_bytes(dst, value.data(), value.size());
}

static uint64_t dtype_size_bytes(const std::string &dtype) {
    if (dtype == "BOOL" || dtype == "I8" || dtype == "U8" ||
        dtype == "F8_E5M2" || dtype == "F8_E4M3" || dtype == "F8_E8M0") {
        return 1;
    }
    if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") {
        return 2;
    }
    if (dtype == "I32" || dtype == "U32" || dtype == "F32") {
        return 4;
    }
    if (dtype == "I64" || dtype == "U64" || dtype == "F64") {
        return 8;
    }
    if (dtype == "F4") {
        return 0;
    }
    throw std::runtime_error("MEGA tensor uses unknown dtype: " + dtype);
}

static uint64_t infer_logical_nbytes(const std::string &dtype, const std::vector<uint64_t> &shape) {
    uint64_t elements = 1;
    for (uint64_t dim : shape) {
        if (dim != 0 && elements > UINT64_MAX / dim) {
            throw std::runtime_error("MEGA tensor element count overflows");
        }
        elements *= dim;
    }
    if (dtype == "F4") {
        if (elements > UINT64_MAX - 1) {
            throw std::runtime_error("MEGA F4 tensor size overflows");
        }
        return (elements + 1) / 2;
    }
    const uint64_t itemsize = dtype_size_bytes(dtype);
    if (itemsize != 0 && elements > UINT64_MAX / itemsize) {
        throw std::runtime_error("MEGA tensor byte size overflows");
    }
    return elements * itemsize;
}

static std::array<uint8_t, SHA256_DIGEST_LENGTH> sha256_bytes(const std::vector<uint8_t> &payload) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) {
        throw std::runtime_error("OpenSSL EVP_MD_CTX_new failed: " + openssl_error_string());
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, payload.data(), payload.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("OpenSSL SHA256 digest failed: " + openssl_error_string());
    }
    std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
    unsigned int digest_len = 0;
    if (EVP_DigestFinal_ex(ctx, digest.data(), &digest_len) != 1 ||
        digest_len != digest.size()) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("OpenSSL SHA256 digest finalization failed: " + openssl_error_string());
    }
    EVP_MD_CTX_free(ctx);
    return digest;
}

static uint32_t crc32_fd_range(
    int fd,
    const std::string &filename,
    uint64_t file_offset,
    uint64_t length
) {
    constexpr uint64_t BUFFER_SIZE = 16ULL * 1024ULL * 1024ULL;
    std::vector<uint8_t> buffer(static_cast<size_t>(std::min<uint64_t>(BUFFER_SIZE, std::max<uint64_t>(length, 1))));
    uint32_t crc = 0;
    uint64_t done = 0;
    while (done < length) {
        const uint64_t step64 = std::min<uint64_t>(length - done, BUFFER_SIZE);
        const size_t step = static_cast<size_t>(step64);
        read_exact_at(fd, file_offset + done, buffer.data(), step64, filename);
        crc = crc32_update(crc, buffer.data(), step);
        done += step64;
    }
    return crc;
}

static std::string openssl_error_string() {
    unsigned long err = ERR_get_error();
    if (err == 0) {
        return "unknown OpenSSL error";
    }
    char buf[256];
    ERR_error_string_n(err, buf, sizeof(buf));
    return std::string(buf);
}

template <typename T, auto FreeFn>
using ossl_ptr = std::unique_ptr<T, decltype(FreeFn)>;

static ossl_ptr<BIO, BIO_free> bio_from_string(const std::string &data) {
    BIO *bio = BIO_new_mem_buf(data.data(), static_cast<int>(data.size()));
    if (bio == nullptr) {
        throw std::runtime_error("OpenSSL BIO_new_mem_buf failed: " + openssl_error_string());
    }
    return ossl_ptr<BIO, BIO_free>(bio, BIO_free);
}

static ossl_ptr<X509, X509_free> read_x509_pem_one(const std::string &pem, const std::string &label) {
    auto bio = bio_from_string(pem);
    X509 *cert = PEM_read_bio_X509(bio.get(), nullptr, nullptr, nullptr);
    if (cert == nullptr) {
        throw std::runtime_error("failed to parse " + label + " certificate PEM: " + openssl_error_string());
    }
    return ossl_ptr<X509, X509_free>(cert, X509_free);
}

static STACK_OF(X509) *read_x509_pem_stack(const std::string &pem, const std::string &label) {
    STACK_OF(X509) *stack = sk_X509_new_null();
    if (stack == nullptr) {
        throw std::runtime_error("failed to allocate " + label + " certificate stack");
    }
    if (pem.empty()) {
        return stack;
    }
    auto bio = bio_from_string(pem);
    while (true) {
        X509 *cert = PEM_read_bio_X509(bio.get(), nullptr, nullptr, nullptr);
        if (cert == nullptr) {
            if (sk_X509_num(stack) == 0) {
                sk_X509_free(stack);
                throw std::runtime_error("failed to parse " + label + " certificate PEM: " + openssl_error_string());
            }
            ERR_clear_error();
            break;
        }
        if (!sk_X509_push(stack, cert)) {
            X509_free(cert);
            sk_X509_pop_free(stack, X509_free);
            throw std::runtime_error("failed to append " + label + " certificate");
        }
    }
    return stack;
}

static void free_x509_stack(STACK_OF(X509) *stack) {
    if (stack != nullptr) {
        sk_X509_pop_free(stack, X509_free);
    }
}

static std::string x509_name_to_string(X509_NAME *name) {
    if (name == nullptr) {
        return "";
    }
    BIO *bio = BIO_new(BIO_s_mem());
    if (bio == nullptr) {
        throw std::runtime_error("OpenSSL BIO_new failed: " + openssl_error_string());
    }
    if (X509_NAME_print_ex(bio, name, 0, XN_FLAG_RFC2253) < 0) {
        BIO_free(bio);
        throw std::runtime_error("OpenSSL X509_NAME_print_ex failed: " + openssl_error_string());
    }
    BUF_MEM *mem = nullptr;
    BIO_get_mem_ptr(bio, &mem);
    std::string out(mem != nullptr && mem->data != nullptr ? mem->data : "", mem != nullptr ? mem->length : 0);
    BIO_free(bio);
    return out;
}

static std::string asn1_time_to_string(const ASN1_TIME *time) {
    BIO *bio = BIO_new(BIO_s_mem());
    if (bio == nullptr) {
        throw std::runtime_error("OpenSSL BIO_new failed: " + openssl_error_string());
    }
    if (ASN1_TIME_print(bio, time) != 1) {
        BIO_free(bio);
        throw std::runtime_error("OpenSSL ASN1_TIME_print failed: " + openssl_error_string());
    }
    BUF_MEM *mem = nullptr;
    BIO_get_mem_ptr(bio, &mem);
    std::string out(mem != nullptr && mem->data != nullptr ? mem->data : "", mem != nullptr ? mem->length : 0);
    BIO_free(bio);
    return out;
}

static std::array<uint8_t, SHA256_DIGEST_LENGTH> sha256_fd_range(
    int fd,
    const std::string &filename,
    uint64_t file_offset,
    uint64_t length
) {
    constexpr uint64_t BUFFER_SIZE = 16ULL * 1024ULL * 1024ULL;
    std::vector<uint8_t> buffer(static_cast<size_t>(std::min<uint64_t>(BUFFER_SIZE, std::max<uint64_t>(length, 1))));
    ossl_ptr<EVP_MD_CTX, EVP_MD_CTX_free> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (ctx == nullptr) {
        throw std::runtime_error("OpenSSL EVP_MD_CTX_new failed: " + openssl_error_string());
    }
    if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("OpenSSL EVP_DigestInit_ex(SHA256) failed: " + openssl_error_string());
    }
    uint64_t done = 0;
    while (done < length) {
        const uint64_t step64 = std::min<uint64_t>(length - done, BUFFER_SIZE);
        read_exact_at(fd, file_offset + done, buffer.data(), step64, filename);
        if (EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<size_t>(step64)) != 1) {
            throw std::runtime_error("OpenSSL EVP_DigestUpdate(SHA256) failed: " + openssl_error_string());
        }
        done += step64;
    }
    std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
    unsigned int digest_len = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &digest_len) != 1) {
        throw std::runtime_error("OpenSSL EVP_DigestFinal_ex(SHA256) failed: " + openssl_error_string());
    }
    if (digest_len != digest.size()) {
        throw std::runtime_error("OpenSSL SHA256 digest length mismatch");
    }
    return digest;
}

static void add_trusted_roots_to_store(X509_STORE *store, STACK_OF(X509) *roots) {
    for (int i = 0; i < sk_X509_num(roots); ++i) {
        X509 *cert = sk_X509_value(roots, i);
        if (X509_STORE_add_cert(store, cert) != 1) {
            const unsigned long err = ERR_peek_last_error();
            if (ERR_GET_REASON(err) == X509_R_CERT_ALREADY_IN_HASH_TABLE) {
                ERR_clear_error();
                continue;
            }
            throw std::runtime_error("failed to add trusted root certificate: " + openssl_error_string());
        }
    }
}

static bool verify_statement_signature(
    X509 *leaf,
    const std::string &statement,
    const std::string &signature,
    const std::string &algorithm,
    std::string &error
) {
    ossl_ptr<EVP_PKEY, EVP_PKEY_free> pubkey(X509_get_pubkey(leaf), EVP_PKEY_free);
    if (pubkey == nullptr) {
        error = "leaf certificate has no public key";
        return false;
    }

    ossl_ptr<EVP_MD_CTX, EVP_MD_CTX_free> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (ctx == nullptr) {
        error = "OpenSSL EVP_MD_CTX_new failed: " + openssl_error_string();
        return false;
    }

    EVP_PKEY_CTX *pkey_ctx = nullptr;
    if (EVP_DigestVerifyInit(ctx.get(), &pkey_ctx, EVP_sha256(), nullptr, pubkey.get()) != 1) {
        error = "OpenSSL EVP_DigestVerifyInit failed: " + openssl_error_string();
        return false;
    }

    const int key_type = EVP_PKEY_base_id(pubkey.get());
    if (algorithm == "sha256-rsa-pss") {
        if (key_type != EVP_PKEY_RSA && key_type != EVP_PKEY_RSA_PSS) {
            error = "signature algorithm requires an RSA public key";
            return false;
        }
        if (EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, RSA_PKCS1_PSS_PADDING) <= 0 ||
            EVP_PKEY_CTX_set_rsa_mgf1_md(pkey_ctx, EVP_sha256()) <= 0 ||
            EVP_PKEY_CTX_set_rsa_pss_saltlen(pkey_ctx, RSA_PSS_SALTLEN_DIGEST) <= 0) {
            error = "OpenSSL RSA-PSS parameter setup failed: " + openssl_error_string();
            return false;
        }
    } else if (algorithm == "sha256-rsa-pkcs1" || algorithm == "sha256-rsa") {
        if (key_type != EVP_PKEY_RSA) {
            error = "signature algorithm requires an RSA public key";
            return false;
        }
        if (EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, RSA_PKCS1_PADDING) <= 0) {
            error = "OpenSSL RSA PKCS#1 parameter setup failed: " + openssl_error_string();
            return false;
        }
    } else if (algorithm == "sha256-ecdsa") {
        if (key_type != EVP_PKEY_EC) {
            error = "signature algorithm requires an EC public key";
            return false;
        }
    } else {
        error = "unsupported signature algorithm: " + algorithm;
        return false;
    }

    if (EVP_DigestVerifyUpdate(ctx.get(), statement.data(), statement.size()) != 1) {
        error = "OpenSSL EVP_DigestVerifyUpdate failed: " + openssl_error_string();
        return false;
    }
    const int rc = EVP_DigestVerifyFinal(
        ctx.get(),
        reinterpret_cast<const unsigned char *>(signature.data()),
        signature.size()
    );
    if (rc == 1) {
        return true;
    }
    if (rc == 0) {
        error = "signature mismatch";
        return false;
    }
    error = "OpenSSL EVP_DigestVerifyFinal failed: " + openssl_error_string();
    return false;
}

static std::string chain_error_to_risk(int err) {
    switch (err) {
        case X509_V_OK:
            return "trusted";
        case X509_V_ERR_CERT_HAS_EXPIRED:
            return "expired";
        case X509_V_ERR_CERT_NOT_YET_VALID:
            return "not_yet_valid";
        default:
            return "untrusted_issuer";
    }
}

static void validate_checksum_field(
    uint32_t checksum_type,
    const std::array<uint8_t, 32> &checksum,
    const std::string &filename,
    const std::string &label
);

static bool leaf_certificate_allows_artifact_signing(X509 *leaf, std::string &error) {
    const uint32_t key_usage = X509_get_key_usage(leaf);
    if (key_usage == UINT32_MAX || (key_usage & KU_DIGITAL_SIGNATURE) == 0) {
        error = "leaf certificate must include keyUsage=digitalSignature";
        return false;
    }
    const uint32_t extended_key_usage = X509_get_extended_key_usage(leaf);
    if (extended_key_usage == UINT32_MAX || (extended_key_usage & XKU_CODE_SIGN) == 0) {
        error = "leaf certificate must include extendedKeyUsage=codeSigning";
        return false;
    }
    return true;
}

static pybind11::dict verify_x509_signature(
    const std::string &leaf_pem,
    const std::string &chain_pem,
    const std::string &trusted_roots_pem,
    const std::string &statement,
    const std::string &signature,
    const std::string &algorithm
) {
    auto leaf = read_x509_pem_one(leaf_pem, "leaf");
    std::unique_ptr<STACK_OF(X509), decltype(&free_x509_stack)> chain(
        read_x509_pem_stack(chain_pem, "intermediate chain"),
        free_x509_stack
    );
    std::unique_ptr<STACK_OF(X509), decltype(&free_x509_stack)> roots(
        read_x509_pem_stack(trusted_roots_pem, "trusted roots"),
        free_x509_stack
    );

    pybind11::dict out;
    out["trusted"] = false;
    out["chain_trusted"] = false;
    out["certificate_policy_valid"] = false;
    out["signature_valid"] = false;
    out["risk"] = pybind11::str("untrusted_issuer");
    out["subject"] = pybind11::str(x509_name_to_string(X509_get_subject_name(leaf.get())));
    out["issuer"] = pybind11::str(x509_name_to_string(X509_get_issuer_name(leaf.get())));
    out["not_before"] = pybind11::str(asn1_time_to_string(X509_get0_notBefore(leaf.get())));
    out["not_after"] = pybind11::str(asn1_time_to_string(X509_get0_notAfter(leaf.get())));
    out["chain_error"] = pybind11::str("");
    out["signature_error"] = pybind11::str("");

    if (sk_X509_num(roots.get()) == 0) {
        out["chain_error"] = pybind11::str("no trusted root certificates supplied");
        return out;
    }

    ossl_ptr<X509_STORE, X509_STORE_free> store(X509_STORE_new(), X509_STORE_free);
    if (store == nullptr) {
        throw std::runtime_error("OpenSSL X509_STORE_new failed: " + openssl_error_string());
    }
    add_trusted_roots_to_store(store.get(), roots.get());

    ossl_ptr<X509_STORE_CTX, X509_STORE_CTX_free> store_ctx(X509_STORE_CTX_new(), X509_STORE_CTX_free);
    if (store_ctx == nullptr) {
        throw std::runtime_error("OpenSSL X509_STORE_CTX_new failed: " + openssl_error_string());
    }
    if (X509_STORE_CTX_init(store_ctx.get(), store.get(), leaf.get(), chain.get()) != 1) {
        throw std::runtime_error("OpenSSL X509_STORE_CTX_init failed: " + openssl_error_string());
    }

    const int chain_rc = X509_verify_cert(store_ctx.get());
    const int chain_err = X509_STORE_CTX_get_error(store_ctx.get());
    if (chain_rc == 1) {
        out["chain_trusted"] = true;
    } else {
        const char *err_str = X509_verify_cert_error_string(chain_err);
        out["risk"] = pybind11::str(chain_error_to_risk(chain_err));
        out["chain_error"] = pybind11::str(err_str != nullptr ? err_str : "certificate chain validation failed");
    }

    std::string cert_policy_error;
    const bool cert_policy_ok = leaf_certificate_allows_artifact_signing(leaf.get(), cert_policy_error);
    out["certificate_policy_valid"] = cert_policy_ok;
    if (!cert_policy_ok) {
        out["risk"] = pybind11::str("certificate_policy_invalid");
        out["certificate_policy_error"] = pybind11::str(cert_policy_error);
    }

    std::string signature_error;
    const bool signature_ok = verify_statement_signature(
        leaf.get(),
        statement,
        signature,
        algorithm,
        signature_error
    );
    out["signature_valid"] = signature_ok;
    if (!signature_ok) {
        out["risk"] = pybind11::str("signature_invalid");
        out["signature_error"] = pybind11::str(signature_error);
    }

    const bool trusted = chain_rc == 1 && cert_policy_ok && signature_ok;
    out["trusted"] = trusted;
    if (trusted) {
        out["risk"] = pybind11::str("trusted");
    }
    return out;
}

static void append_metadata_value(std::vector<uint8_t> &out, pybind11::handle value) {
    if (pybind11::isinstance<pybind11::str>(value)) {
        append_le<uint32_t>(out, MEGA_META_STRING);
        append_string(out, pybind11::cast<std::string>(value));
        return;
    }
    if (pybind11::isinstance<pybind11::bool_>(value)) {
        append_le<uint32_t>(out, MEGA_META_BOOL);
        append_le<uint8_t>(out, pybind11::cast<bool>(value) ? 1 : 0);
        return;
    }
    if (pybind11::isinstance<pybind11::int_>(value)) {
        const uint64_t v = pybind11::cast<uint64_t>(value);
        if (v <= std::numeric_limits<uint32_t>::max()) {
            append_le<uint32_t>(out, MEGA_META_UINT32);
            append_le<uint32_t>(out, static_cast<uint32_t>(v));
        } else {
            append_le<uint32_t>(out, MEGA_META_UINT64);
            append_le<uint64_t>(out, v);
        }
        return;
    }
    if (pybind11::isinstance<pybind11::list>(value) ||
        pybind11::isinstance<pybind11::tuple>(value)) {
        pybind11::sequence seq = pybind11::reinterpret_borrow<pybind11::sequence>(value);
        append_le<uint32_t>(out, MEGA_META_ARRAY);
        const size_t n = pybind11::len(seq);
        if (n == 0 || pybind11::isinstance<pybind11::str>(seq[0])) {
            append_le<uint32_t>(out, MEGA_META_STRING);
            append_le<uint64_t>(out, static_cast<uint64_t>(n));
            for (pybind11::handle item : seq) {
                append_string(out, pybind11::cast<std::string>(item));
            }
            return;
        }
        if (pybind11::isinstance<pybind11::int_>(seq[0])) {
            append_le<uint32_t>(out, MEGA_META_UINT32);
            append_le<uint64_t>(out, static_cast<uint64_t>(n));
            for (pybind11::handle item : seq) {
                const uint64_t v = pybind11::cast<uint64_t>(item);
                if (v > std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("MEGAKV metadata array integer exceeds UINT32");
                }
                append_le<uint32_t>(out, static_cast<uint32_t>(v));
            }
            return;
        }
    }
    throw std::runtime_error("unsupported MEGAKV metadata value type");
}

static void append_metadata_entries(std::vector<uint8_t> &out, pybind11::dict metadata) {
    for (auto item : metadata) {
        append_string(out, pybind11::cast<std::string>(item.first));
        append_metadata_value(out, item.second);
    }
}

static void write_all_fd(int fd, const uint8_t *data, size_t size, const std::string &filename) {
    size_t done = 0;
    while (done < size) {
        const ssize_t n = write(fd, data + done, size - done);
        if (n <= 0) {
            throw std::runtime_error(filename + ": failed to write MEGAKV file");
        }
        done += static_cast<size_t>(n);
    }
}

static void write_all_fd(int fd, const std::vector<uint8_t> &data, const std::string &filename) {
    if (!data.empty()) {
        write_all_fd(fd, data.data(), data.size(), filename);
    }
}

static void write_kv_file(
    const std::string &filename,
    pybind11::iterable entries,
    pybind11::dict metadata,
    uint64_t alignment
) {
    if (alignment == 0) {
        throw std::runtime_error("MEGAKV alignment must be non-zero");
    }
    std::vector<pybind11::dict> entry_dicts;
    for (pybind11::handle item : entries) {
        entry_dicts.push_back(pybind11::reinterpret_borrow<pybind11::dict>(item));
    }

    std::vector<uint8_t> header;
    append_le<uint32_t>(header, MEGAKV_MAGIC);
    append_le<uint32_t>(header, 1);
    append_le<uint64_t>(header, static_cast<uint64_t>(entry_dicts.size()));
    append_le<uint64_t>(header, static_cast<uint64_t>(metadata.size()));
    append_metadata_entries(header, metadata);

    std::vector<uint8_t> payload;
    for (size_t entry_id = 0; entry_id < entry_dicts.size(); ++entry_id) {
        pybind11::dict entry = entry_dicts[entry_id];
        const std::string name = entry.contains("name")
            ? entry["name"].cast<std::string>()
            : "kv." + std::to_string(entry_id);
        const std::string dtype = entry["logical_dtype"].cast<std::string>();
        const std::string data_bytes = entry["data"].cast<std::string>();
        std::vector<uint8_t> data(data_bytes.begin(), data_bytes.end());
        const uint64_t payload_offset = static_cast<uint64_t>(payload.size());
        payload.insert(payload.end(), data.begin(), data.end());

        pybind11::sequence shape_seq = pybind11::reinterpret_borrow<pybind11::sequence>(entry["shape"]);
        std::vector<uint64_t> shape;
        shape.reserve(pybind11::len(shape_seq));
        for (pybind11::handle dim : shape_seq) {
            shape.push_back(pybind11::cast<uint64_t>(dim));
        }

        const uint64_t logical_nbytes = entry.contains("logical_nbytes")
            ? entry["logical_nbytes"].cast<uint64_t>()
            : static_cast<uint64_t>(data.size());
        const uint64_t stored_nbytes = entry.contains("stored_nbytes")
            ? entry["stored_nbytes"].cast<uint64_t>()
            : static_cast<uint64_t>(data.size());
        if (stored_nbytes != data.size()) {
            throw std::runtime_error("MEGAKV stored_nbytes must match data length");
        }
        const uint32_t checksum_type = entry.contains("checksum_type")
            ? entry["checksum_type"].cast<uint32_t>()
            : MEGA_CHECKSUM_SHA256;
        std::array<uint8_t, 32> checksum;
        checksum.fill(0);
        if (entry.contains("checksum")) {
            const std::string checksum_bytes = entry["checksum"].cast<std::string>();
            if (checksum_bytes.size() != checksum.size()) {
                throw std::runtime_error("MEGAKV checksum must be exactly 32 bytes");
            }
            std::memcpy(checksum.data(), checksum_bytes.data(), checksum.size());
            validate_checksum_field(checksum_type, checksum, filename, "MEGAKV entry");
        } else if (checksum_type == MEGA_CHECKSUM_CRC32) {
            const uint32_t crc = crc32_bytes(data);
            std::memcpy(checksum.data(), &crc, sizeof(crc));
        } else if (checksum_type == MEGA_CHECKSUM_SHA256) {
            const auto digest = sha256_bytes(data);
            std::copy(digest.begin(), digest.end(), checksum.begin());
        } else if (checksum_type != MEGA_CHECKSUM_NONE) {
            throw std::runtime_error("unsupported MEGAKV checksum type");
        }

        append_string(header, name);
        append_le<uint32_t>(header, entry.contains("layer_id") ? entry["layer_id"].cast<uint32_t>() : 0);
        append_le<uint32_t>(header, entry.contains("kv_role") ? entry["kv_role"].cast<uint32_t>() : 0);
        append_le<uint64_t>(header, entry.contains("sequence_id") ? entry["sequence_id"].cast<uint64_t>() : 0);
        append_le<uint64_t>(header, entry.contains("token_start") ? entry["token_start"].cast<uint64_t>() : 0);
        append_le<uint64_t>(header, entry.contains("token_count") ? entry["token_count"].cast<uint64_t>() : 0);
        append_le<uint32_t>(header, static_cast<uint32_t>(shape.size()));
        for (uint64_t dim : shape) {
            append_le<uint64_t>(header, dim);
        }
        append_string(header, dtype);
        append_le<uint64_t>(header, payload_offset);
        append_le<uint64_t>(header, logical_nbytes);
        append_le<uint64_t>(header, stored_nbytes);
        append_le<uint32_t>(header, entry.contains("flags") ? entry["flags"].cast<uint32_t>() : 0);
        append_le<uint32_t>(header, entry.contains("codec") ? entry["codec"].cast<uint32_t>() : MEGA_COMPRESSION_NONE);
        append_le<uint32_t>(header, entry.contains("shuffle_elem_size") ? entry["shuffle_elem_size"].cast<uint32_t>() : 0);
        append_le<uint32_t>(header, 0);
        append_le<uint32_t>(header, checksum_type);
        append_bytes(header, checksum.data(), checksum.size());
    }

    const uint64_t pad = (alignment - (header.size() % alignment)) % alignment;
    pybind11::gil_scoped_release release;
    const int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC
#ifdef _MSC_VER
        | _O_BINARY
#endif
        , 0644);
    if (fd < 0) {
        throw std::runtime_error(filename + ": failed to open MEGAKV file for writing");
    }
    try {
        write_all_fd(fd, header, filename);
        if (pad > 0) {
            std::vector<uint8_t> zeros(static_cast<size_t>(pad), 0);
            write_all_fd(fd, zeros, filename);
        }
        write_all_fd(fd, payload, filename);
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
}

static void validate_checksum_field(
    uint32_t checksum_type,
    const std::array<uint8_t, 32> &checksum,
    const std::string &filename,
    const std::string &label
) {
    switch (checksum_type) {
        case MEGA_CHECKSUM_NONE:
            if (!checksum_tail_is_zero(checksum, 0)) {
                throw std::runtime_error(filename + ": checksum-less " + label + " must use a zero checksum field");
            }
            return;
        case MEGA_CHECKSUM_CRC32:
            if (!checksum_tail_is_zero(checksum, 4)) {
                throw std::runtime_error(filename + ": CRC32 " + label + " checksum must use only the first 4 bytes");
            }
            return;
        case MEGA_CHECKSUM_SHA256:
            return;
        default:
            throw std::runtime_error(filename + ": " + label + " uses unknown checksum type");
    }
}

static void verify_stored_payload_checksum(
    const std::vector<uint8_t> &payload,
    uint32_t checksum_type,
    const std::array<uint8_t, 32> &checksum,
    const std::string &filename,
    const std::string &tensor_name,
    const std::string &label
) {
    validate_checksum_field(checksum_type, checksum, filename, label);
    if (checksum_type == MEGA_CHECKSUM_NONE) {
        return;
    }
    if (checksum_type == MEGA_CHECKSUM_CRC32) {
        const uint32_t expected = read_checksum_u32_le(checksum);
        const uint32_t actual = crc32_bytes(payload);
        if (actual != expected) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " " + label + " checksum mismatch");
        }
        return;
    }
    if (checksum_type == MEGA_CHECKSUM_SHA256) {
        const auto actual = sha256_bytes(payload);
        if (!std::equal(actual.begin(), actual.end(), checksum.begin())) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " " + label + " checksum mismatch");
        }
        return;
    }
}

struct mega_file_tensor_source {
    std::string name;
    std::string dtype;
    std::string storage_format;
    std::vector<uint64_t> shape;
    uint64_t payload_offset;
    uint64_t logical_nbytes;
    uint64_t stored_nbytes;
    uint32_t tensor_flags;
    uint32_t compression_codec;
    uint32_t shuffle_elem_size;
    uint32_t checksum_type;
    std::array<uint8_t, 32> checksum;
    std::string src_filename;
    uint64_t src_offset;
};

static void copy_fd_range_buffered(
    int src_fd,
    int dst_fd,
    uint64_t src_offset,
    uint64_t length,
    const std::string &filename
) {
    constexpr uint64_t BUFFER_SIZE = 16ULL * 1024ULL * 1024ULL;
    std::vector<uint8_t> buffer(static_cast<size_t>(std::min<uint64_t>(BUFFER_SIZE, std::max<uint64_t>(length, 1))));
    uint64_t done = 0;
    while (done < length) {
        const uint64_t step = std::min<uint64_t>(length - done, BUFFER_SIZE);
        read_exact_at(src_fd, src_offset + done, buffer.data(), step, filename);
        write_all_fd(dst_fd, buffer.data(), static_cast<size_t>(step), filename);
        done += step;
    }
}

static bool copy_fd_range_sendfile(
    int src_fd,
    int dst_fd,
    uint64_t src_offset,
    uint64_t length
) {
#ifdef _MSC_VER
    (void)src_fd;
    (void)dst_fd;
    (void)src_offset;
    (void)length;
    return false;
#else
    if (length == 0) {
        return true;
    }
    uint64_t done = 0;
    off_t offset = static_cast<off_t>(src_offset);
    while (done < length) {
        const size_t step = static_cast<size_t>(
            std::min<uint64_t>(length - done, static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()))
        );
        const ssize_t n = sendfile(dst_fd, src_fd, &offset, step);
        if (n > 0) {
            done += static_cast<uint64_t>(n);
            continue;
        }
        if (n == 0) {
            throw std::runtime_error("source file ended during MEGA payload copy");
        }
        if (errno == EINTR || errno == EAGAIN) {
            continue;
        }
        return false;
    }
    return true;
#endif
}

static void copy_fd_range(
    int src_fd,
    int dst_fd,
    uint64_t src_offset,
    uint64_t length,
    const std::string &filename
) {
    if (copy_fd_range_sendfile(src_fd, dst_fd, src_offset, length)) {
        return;
    }
    copy_fd_range_buffered(src_fd, dst_fd, src_offset, length, filename);
}

static pybind11::list encode_file_tensors(
    const std::string &payload_filename,
    pybind11::list tensors,
    uint32_t compression_codec,
    int compression_level,
    double min_ratio
) {
    if (compression_codec != MEGA_COMPRESSION_AUTO && compression_codec > MEGA_COMPRESSION_ZSTD) {
        throw std::runtime_error("MEGA compression codec is unknown");
    }
    if (min_ratio <= 0.0 || min_ratio > 1.0) {
        throw std::runtime_error("MEGA compression min_ratio must be in (0, 1]");
    }
    std::vector<mega_file_tensor_source> normalized;
    normalized.reserve(static_cast<size_t>(pybind11::len(tensors)));
    for (pybind11::handle item : tensors) {
        pybind11::dict tensor = pybind11::reinterpret_borrow<pybind11::dict>(item);
        mega_file_tensor_source rec;
        rec.name = tensor["name"].cast<std::string>();
        rec.dtype = tensor["logical_dtype"].cast<std::string>();
        rec.storage_format = tensor.contains("storage_format")
            ? tensor["storage_format"].cast<std::string>()
            : "raw_dense";
        pybind11::sequence shape_seq = pybind11::reinterpret_borrow<pybind11::sequence>(tensor["shape"]);
        rec.shape.reserve(pybind11::len(shape_seq));
        for (pybind11::handle dim : shape_seq) {
            rec.shape.push_back(pybind11::cast<uint64_t>(dim));
        }
        rec.payload_offset = 0;
        rec.logical_nbytes = tensor["logical_nbytes"].cast<uint64_t>();
        rec.stored_nbytes = tensor["stored_nbytes"].cast<uint64_t>();
        rec.tensor_flags = 0;
        rec.compression_codec = MEGA_COMPRESSION_NONE;
        rec.shuffle_elem_size = 0;
        rec.checksum_type = MEGA_CHECKSUM_NONE;
        rec.checksum.fill(0);
        rec.src_filename = tensor["src_filename"].cast<std::string>();
        rec.src_offset = tensor["src_offset"].cast<uint64_t>();
        normalized.push_back(std::move(rec));
    }

    std::vector<mega_file_tensor_source> out_records;
    out_records.reserve(normalized.size());
    {
        pybind11::gil_scoped_release release;
        const int dst_fd = open(payload_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC
#ifdef _MSC_VER
            | _O_BINARY
#endif
            , 0644);
        if (dst_fd < 0) {
            throw std::runtime_error(payload_filename + ": failed to open encoded MEGA payload file");
        }
        std::unordered_map<std::string, int> src_fds;
        uint64_t payload_offset = 0;
        try {
            for (auto rec : normalized) {
                auto it = src_fds.find(rec.src_filename);
                if (it == src_fds.end()) {
                    const int src_fd = open(rec.src_filename.c_str(), O_RDONLY
#ifdef _MSC_VER
                        | _O_BINARY
#endif
                    , 0644);
                    if (src_fd < 0) {
                        throw std::runtime_error(rec.src_filename + ": failed to open tensor source file");
                    }
                    it = src_fds.emplace(rec.src_filename, src_fd).first;
                }
                std::vector<uint8_t> raw(static_cast<size_t>(rec.logical_nbytes));
                read_exact_at(it->second, rec.src_offset, raw.data(), rec.logical_nbytes, rec.src_filename);
                const bool try_compress = compression_codec != MEGA_COMPRESSION_NONE && rec.logical_nbytes > 0;
                std::vector<uint8_t> encoded;
                uint32_t selected_codec = compression_codec;
                if (try_compress) {
                    if (compression_codec == MEGA_COMPRESSION_AUTO) {
                        encoded = compress_payload(MEGA_COMPRESSION_ZSTD, raw, compression_level);
                        selected_codec = MEGA_COMPRESSION_ZSTD;
                    } else {
                        encoded = compress_payload(compression_codec, raw, compression_level);
                    }
                }
                const bool use_compressed =
                    try_compress &&
                    encoded.size() < raw.size() &&
                    static_cast<double>(encoded.size()) / static_cast<double>(raw.size()) <= min_ratio;
                const std::vector<uint8_t> &payload = use_compressed ? encoded : raw;
                rec.payload_offset = payload_offset;
                rec.stored_nbytes = static_cast<uint64_t>(payload.size());
                rec.tensor_flags = use_compressed ? MEGA_TENSOR_FLAG_COMPRESSED : 0;
                rec.compression_codec = use_compressed ? selected_codec : MEGA_COMPRESSION_NONE;
                write_all_fd(dst_fd, payload, payload_filename);
                payload_offset += rec.stored_nbytes;
                out_records.push_back(std::move(rec));
            }
        } catch (...) {
            for (auto &item : src_fds) {
                close(item.second);
            }
            close(dst_fd);
            throw;
        }
        for (auto &item : src_fds) {
            close(item.second);
        }
        close(dst_fd);
    }

    pybind11::list out;
    for (const auto &rec : out_records) {
        pybind11::dict item;
        item["name"] = rec.name;
        pybind11::list shape;
        for (uint64_t dim : rec.shape) {
            shape.append(dim);
        }
        item["shape"] = shape;
        item["logical_dtype"] = rec.dtype;
        item["storage_format"] = rec.storage_format;
        item["payload_offset"] = rec.payload_offset;
        item["logical_nbytes"] = rec.logical_nbytes;
        item["stored_nbytes"] = rec.stored_nbytes;
        item["tensor_flags"] = rec.tensor_flags;
        item["compression_codec"] = rec.compression_codec;
        item["shuffle_elem_size"] = rec.shuffle_elem_size;
        item["src_filename"] = payload_filename;
        item["src_offset"] = rec.payload_offset;
        out.append(item);
    }
    return out;
}

static void write_file(
    const std::string &dst_filename,
    pybind11::list tensors,
    pybind11::dict metadata,
    uint64_t alignment
) {
    if (alignment == 0) {
        throw std::runtime_error("MEGA alignment must be non-zero");
    }
    std::vector<mega_file_tensor_source> normalized;
    normalized.reserve(static_cast<size_t>(pybind11::len(tensors)));

    std::vector<uint8_t> header;
    append_le<uint32_t>(header, MEGA_MAGIC);
    append_le<uint32_t>(header, MEGA_FORMAT_VERSION);
    append_le<uint64_t>(header, static_cast<uint64_t>(pybind11::len(tensors)));
    append_le<uint64_t>(header, static_cast<uint64_t>(metadata.size()));
    append_metadata_entries(header, metadata);

    for (pybind11::handle item : tensors) {
        pybind11::dict tensor = pybind11::reinterpret_borrow<pybind11::dict>(item);
        mega_file_tensor_source rec;
        rec.name = tensor["name"].cast<std::string>();
        rec.dtype = tensor["logical_dtype"].cast<std::string>();
        rec.storage_format = tensor.contains("storage_format")
            ? tensor["storage_format"].cast<std::string>()
            : "raw_dense";
        pybind11::sequence shape_seq = pybind11::reinterpret_borrow<pybind11::sequence>(tensor["shape"]);
        rec.shape.reserve(pybind11::len(shape_seq));
        for (pybind11::handle dim : shape_seq) {
            rec.shape.push_back(pybind11::cast<uint64_t>(dim));
        }
        rec.payload_offset = tensor["payload_offset"].cast<uint64_t>();
        rec.logical_nbytes = tensor.contains("logical_nbytes")
            ? tensor["logical_nbytes"].cast<uint64_t>()
            : infer_logical_nbytes(rec.dtype, rec.shape);
        const uint64_t inferred_logical_nbytes = infer_logical_nbytes(rec.dtype, rec.shape);
        if (rec.logical_nbytes != inferred_logical_nbytes) {
            throw std::runtime_error("MEGA tensor " + rec.name + " logical_nbytes does not match dtype and shape");
        }
        rec.stored_nbytes = tensor.contains("stored_nbytes")
            ? tensor["stored_nbytes"].cast<uint64_t>()
            : rec.logical_nbytes;
        rec.tensor_flags = tensor.contains("tensor_flags") ? tensor["tensor_flags"].cast<uint32_t>() : 0;
        rec.compression_codec = tensor.contains("compression_codec") ? tensor["compression_codec"].cast<uint32_t>() : MEGA_COMPRESSION_NONE;
        rec.shuffle_elem_size = tensor.contains("shuffle_elem_size") ? tensor["shuffle_elem_size"].cast<uint32_t>() : 0;
        rec.checksum_type = tensor.contains("checksum_type") ? tensor["checksum_type"].cast<uint32_t>() : MEGA_CHECKSUM_NONE;
        rec.checksum.fill(0);
        if (tensor.contains("checksum")) {
            const std::string checksum_bytes = tensor["checksum"].cast<std::string>();
            if (checksum_bytes.size() != rec.checksum.size()) {
                throw std::runtime_error("MEGA tensor checksum must be exactly 32 bytes");
            }
            std::memcpy(rec.checksum.data(), checksum_bytes.data(), rec.checksum.size());
        }
        validate_checksum_field(rec.checksum_type, rec.checksum, dst_filename, "tensor " + rec.name);
        rec.src_filename = tensor["src_filename"].cast<std::string>();
        rec.src_offset = tensor["src_offset"].cast<uint64_t>();

        uint32_t dir_flags = 0;
        if (rec.storage_format != "raw_dense") {
            dir_flags |= MEGA_TENSOR_DIR_HAS_STORAGE_FORMAT;
        }
        if (rec.stored_nbytes != rec.logical_nbytes) {
            dir_flags |= MEGA_TENSOR_DIR_HAS_STORED_NBYTES;
        }
        if (rec.tensor_flags != 0) {
            dir_flags |= MEGA_TENSOR_DIR_HAS_TENSOR_FLAGS;
        }
        if (rec.compression_codec != MEGA_COMPRESSION_NONE) {
            dir_flags |= MEGA_TENSOR_DIR_HAS_COMPRESSION_CODEC;
        }
        if (rec.shuffle_elem_size != 0) {
            dir_flags |= MEGA_TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE;
        }
        if (rec.checksum_type != MEGA_CHECKSUM_NONE) {
            dir_flags |= MEGA_TENSOR_DIR_HAS_CHECKSUM;
        }

        append_compact_string(header, rec.name);
        append_le<uint32_t>(header, dir_flags);
        append_le<uint32_t>(header, static_cast<uint32_t>(rec.shape.size()));
        for (auto it = rec.shape.rbegin(); it != rec.shape.rend(); ++it) {
            append_le<uint64_t>(header, *it);
        }
        append_compact_string(header, rec.dtype);
        append_le<uint64_t>(header, rec.payload_offset);
        if ((dir_flags & MEGA_TENSOR_DIR_HAS_STORAGE_FORMAT) != 0) {
            append_compact_string(header, rec.storage_format);
        }
        if ((dir_flags & MEGA_TENSOR_DIR_HAS_STORED_NBYTES) != 0) {
            append_le<uint64_t>(header, rec.stored_nbytes);
        }
        if ((dir_flags & MEGA_TENSOR_DIR_HAS_TENSOR_FLAGS) != 0) {
            append_le<uint32_t>(header, rec.tensor_flags);
        }
        if ((dir_flags & MEGA_TENSOR_DIR_HAS_COMPRESSION_CODEC) != 0) {
            append_le<uint32_t>(header, rec.compression_codec);
        }
        if ((dir_flags & MEGA_TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE) != 0) {
            append_le<uint32_t>(header, rec.shuffle_elem_size);
        }
        if ((dir_flags & MEGA_TENSOR_DIR_HAS_CHECKSUM) != 0) {
            append_le<uint32_t>(header, rec.checksum_type);
            append_bytes(header, rec.checksum.data(), rec.checksum.size());
        }
        normalized.push_back(std::move(rec));
    }

    const uint64_t pad = (alignment - (header.size() % alignment)) % alignment;
    pybind11::gil_scoped_release release;
    const int dst_fd = open(dst_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC
#ifdef _MSC_VER
        | _O_BINARY
#endif
        , 0644);
    if (dst_fd < 0) {
        throw std::runtime_error(dst_filename + ": failed to open MEGA file for writing");
    }
    try {
        write_all_fd(dst_fd, header, dst_filename);
        if (pad > 0) {
            std::vector<uint8_t> zeros(static_cast<size_t>(pad), 0);
            write_all_fd(dst_fd, zeros, dst_filename);
        }
        std::unordered_map<std::string, int> src_fds;
        try {
            for (const auto &rec : normalized) {
                auto it = src_fds.find(rec.src_filename);
                if (it == src_fds.end()) {
                    const int src_fd = open(rec.src_filename.c_str(), O_RDONLY
#ifdef _MSC_VER
                        | _O_BINARY
#endif
                    , 0644);
                    if (src_fd < 0) {
                        throw std::runtime_error(rec.src_filename + ": failed to open tensor source file");
                    }
                    it = src_fds.emplace(rec.src_filename, src_fd).first;
                }
                copy_fd_range(
                    it->second,
                    dst_fd,
                    rec.src_offset,
                    rec.stored_nbytes,
                    rec.src_filename
                );
            }
        } catch (...) {
            for (auto &item : src_fds) {
                close(item.second);
            }
            throw;
        }
        for (auto &item : src_fds) {
            close(item.second);
        }
    } catch (...) {
        close(dst_fd);
        throw;
    }
    close(dst_fd);
}

static void unshuffle_payload_inplace(std::vector<uint8_t> &payload, uint32_t elem_size) {
    if (elem_size == 0) {
        throw std::runtime_error("MEGA byte-shuffled payload has zero element size");
    }
    if (payload.size() % elem_size != 0) {
        throw std::runtime_error("MEGA byte-shuffled payload length is not divisible by element size");
    }
    if (elem_size == 1 || payload.empty()) {
        return;
    }
    const size_t element_count = payload.size() / elem_size;
    std::vector<uint8_t> out(payload.size());
    for (uint32_t byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
        const size_t shuffled_base = static_cast<size_t>(byte_idx) * element_count;
        for (size_t element_idx = 0; element_idx < element_count; ++element_idx) {
            out[element_idx * elem_size + byte_idx] = payload[shuffled_base + element_idx];
        }
    }
    payload.swap(out);
}

static void copy_decoded_payload_to_dst(uintptr_t dst_ptr, bool dst_is_cuda, const std::vector<uint8_t> &payload) {
    if (payload.empty()) {
        return;
    }
    if (dst_ptr == 0) {
        throw std::runtime_error("MEGA decode destination pointer is null");
    }
    if (dst_is_cuda) {
        if (!cuda_fns.cudaMemcpy || !cuda_fns.cudaDeviceSynchronize) {
            throw std::runtime_error("MEGA decode destination is GPU but GPU runtime is not loaded");
        }
        const cudaError_t err = cuda_fns.cudaMemcpy(
            reinterpret_cast<void *>(dst_ptr),
            payload.data(),
            payload.size(),
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            throw std::runtime_error("MEGA decode cudaMemcpy HostToDevice failed");
        }
        const cudaError_t sync_err = cuda_fns.cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            throw std::runtime_error("MEGA decode cudaDeviceSynchronize failed");
        }
    } else {
        std::memcpy(reinterpret_cast<void *>(dst_ptr), payload.data(), payload.size());
    }
}

static std::vector<uint8_t> read_decode_payload(
    int fd,
    const std::string &filename,
    const std::string &tensor_name,
    uint64_t file_offset,
    uint64_t stored_nbytes,
    uint64_t logical_nbytes,
    uint32_t tensor_flags,
    uint32_t compression_codec,
    uint32_t shuffle_elem_size,
    uint32_t checksum_type,
    const std::array<uint8_t, 32> &checksum,
    const std::string &checksum_label
) {
    if ((tensor_flags & ~(MEGA_TENSOR_FLAG_COMPRESSED | MEGA_TENSOR_FLAG_BYTE_SHUFFLED)) != 0) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " has unknown tensor flags");
    }
    const bool compressed = (tensor_flags & MEGA_TENSOR_FLAG_COMPRESSED) != 0;
    const bool byte_shuffled = (tensor_flags & MEGA_TENSOR_FLAG_BYTE_SHUFFLED) != 0;
    if (!compressed && compression_codec != MEGA_COMPRESSION_NONE) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " is uncompressed but has a non-NONE codec");
    }
    if (byte_shuffled && shuffle_elem_size == 0) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " has zero shuffle element size");
    }
    if (logical_nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        stored_nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " payload is too large for this platform");
    }

    std::vector<uint8_t> payload(static_cast<size_t>(stored_nbytes));
    read_exact_at(fd, file_offset, payload.data(), stored_nbytes, filename);
    verify_stored_payload_checksum(payload, checksum_type, checksum, filename, tensor_name, checksum_label);
    if (compressed) {
        payload = decompress_payload(compression_codec, payload, logical_nbytes);
    } else if (payload.size() != static_cast<size_t>(logical_nbytes)) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " stored size differs from logical size");
    }
    if (payload.size() != static_cast<size_t>(logical_nbytes)) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " decoded size differs from logical size");
    }
    if (byte_shuffled) {
        unshuffle_payload_inplace(payload, shuffle_elem_size);
    }
    if (payload.size() != static_cast<size_t>(logical_nbytes)) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " final decoded size differs from logical size");
    }
    return payload;
}

static void decode_payload_fd(
    int fd,
    const std::string &filename,
    const std::string &tensor_name,
    uint64_t file_offset,
    uint64_t stored_nbytes,
    uint64_t logical_nbytes,
    uint32_t tensor_flags,
    uint32_t compression_codec,
    uint32_t shuffle_elem_size,
    uint32_t checksum_type,
    const std::array<uint8_t, 32> &checksum,
    uintptr_t dst_ptr,
    bool dst_is_cuda
) {
    std::vector<uint8_t> payload = read_decode_payload(
        fd,
        filename,
        tensor_name,
        file_offset,
        stored_nbytes,
        logical_nbytes,
        tensor_flags,
        compression_codec,
        shuffle_elem_size,
        checksum_type,
        checksum,
        "payload"
    );
    copy_decoded_payload_to_dst(dst_ptr, dst_is_cuda, payload);
}

static void validate_chunk_records(
    const std::vector<mega_chunk_record> &chunks,
    uint64_t logical_nbytes,
    uint32_t shuffle_elem_size,
    const std::string &filename,
    const std::string &tensor_name
) {
    uint64_t next = 0;
    for (const auto &chunk : chunks) {
        if (chunk.logical_offset != next) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " has non-contiguous chunk logical ranges");
        }
        if (chunk.logical_size == 0) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " has zero-sized chunk");
        }
        if (chunk.logical_size > UINT64_MAX - next) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " chunk logical range overflows");
        }
        if ((chunk.flags & MEGA_TENSOR_FLAG_BYTE_SHUFFLED) != 0 &&
            (shuffle_elem_size == 0 || chunk.logical_size % shuffle_elem_size != 0)) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " chunk has incompatible shuffle size");
        }
        next += chunk.logical_size;
    }
    if (next != logical_nbytes) {
        throw std::runtime_error(filename + ": tensor " + tensor_name + " chunks do not cover logical tensor bytes");
    }
}

static void decode_chunks_fd(
    int fd,
    const std::string &filename,
    const std::string &tensor_name,
    uint64_t header_length,
    const std::vector<mega_chunk_record> &chunks,
    uint64_t logical_nbytes,
    uint32_t shuffle_elem_size,
    uintptr_t dst_ptr,
    bool dst_is_cuda
) {
    validate_chunk_records(chunks, logical_nbytes, shuffle_elem_size, filename, tensor_name);
    for (const auto &chunk : chunks) {
        std::vector<uint8_t> payload = read_decode_payload(
            fd,
            filename,
            tensor_name,
            header_length + chunk.payload_offset,
            chunk.stored_size,
            chunk.logical_size,
            chunk.flags,
            chunk.codec,
            shuffle_elem_size,
            chunk.checksum_type,
            chunk.checksum,
            "chunk " + std::to_string(chunk.chunk_id)
        );
        if (chunk.logical_offset > UINTPTR_MAX - dst_ptr) {
            throw std::runtime_error(filename + ": tensor " + tensor_name + " destination pointer overflows");
        }
        copy_decoded_payload_to_dst(dst_ptr + static_cast<uintptr_t>(chunk.logical_offset), dst_is_cuda, payload);
    }
}

struct mega_tensor_dir_entry {
    std::string name;
    std::string logical_dtype;
    std::string storage_format;
    std::vector<uint64_t> shape;
    uint64_t payload_offset;
    uint64_t logical_nbytes;
    uint64_t stored_nbytes;
    uint32_t tensor_flags;
    uint32_t compression_codec;
    uint32_t shuffle_elem_size;
    uint32_t checksum_type;
    std::array<uint8_t, 32> checksum;
};

class binary_metadata_parser {
public:
    binary_metadata_parser(const uint8_t *data, uint64_t size, const std::string &filename, const std::string &format_name)
        : _data(data), _fd(-1), _size(size), _filename(filename), _format_name(format_name), _pos(0) {}

    binary_metadata_parser(int fd, uint64_t size, const std::string &filename, const std::string &format_name)
        : _data(nullptr), _fd(fd), _size(size), _filename(filename), _format_name(format_name), _pos(0) {}

    pybind11::dict parse_footer_metadata() {
        const uint32_t magic = read_u32();
        if (magic != MEGA_FOOTER_MAGIC) {
            throw std::runtime_error(_filename + ": invalid MEGA footer overlay magic");
        }
        const uint64_t kv_count = read_u64();
        pybind11::dict metadata = read_metadata(kv_count, "footer overlay metadata");
        if (_pos != _size) {
            throw std::runtime_error(_filename + ": MEGA footer overlay has trailing bytes");
        }
        return metadata;
    }

protected:
    const uint8_t *_data;
    int _fd;
    uint64_t _size;
    std::string _filename;
    std::string _format_name;
    uint64_t _pos;

    static uint64_t align_up(uint64_t value, uint64_t alignment, const std::string &label) {
        const uint64_t rem = value % alignment;
        if (rem == 0) return value;
        if (value > UINT64_MAX - (alignment - rem)) {
            throw std::runtime_error(label + " alignment overflow");
        }
        return value + (alignment - rem);
    }

    void require(uint64_t length) {
        if (length > _size || _pos > _size - length) {
            throw std::runtime_error(_filename + ": unexpected EOF while parsing " + _format_name);
        }
    }

    void read_bytes(void *dst, uint64_t length) {
        require(length);
        if (_data != nullptr) {
            std::memcpy(dst, _data + _pos, static_cast<size_t>(length));
            _pos += length;
            return;
        }
        uint64_t done = 0;
        while (done < length) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(length - done, 1ULL << 30));
            const int64_t n = pread(_fd, reinterpret_cast<uint8_t *>(dst) + done, chunk, _pos + done);
            if (n <= 0) {
                throw std::runtime_error(_filename + ": failed to read " + _format_name);
            }
            done += static_cast<uint64_t>(n);
        }
        _pos += length;
    }

    void read_bytes_at(uint64_t offset, void *dst, uint64_t length) {
        if (length > _size || offset > _size - length) {
            throw std::runtime_error(_filename + ": unexpected EOF while reading " + _format_name);
        }
        if (_data != nullptr) {
            std::memcpy(dst, _data + offset, static_cast<size_t>(length));
            return;
        }
        uint64_t done = 0;
        while (done < length) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(length - done, 1ULL << 30));
            const int64_t n = pread(_fd, reinterpret_cast<uint8_t *>(dst) + done, chunk, offset + done);
            if (n <= 0) {
                throw std::runtime_error(_filename + ": failed to read " + _format_name);
            }
            done += static_cast<uint64_t>(n);
        }
    }

    template <typename T>
    T read_le() {
        T value = 0;
        read_bytes(&value, sizeof(T));
        return value;
    }

    template <typename T>
    T read_le_at(uint64_t offset) {
        T value = 0;
        read_bytes_at(offset, &value, sizeof(T));
        return value;
    }

    uint32_t read_u32() { return read_le<uint32_t>(); }
    uint64_t read_u64() { return read_le<uint64_t>(); }

    std::string read_string() {
        const uint64_t length = read_u64();
        if (length > SIZE_MAX) {
            throw std::runtime_error(_filename + ": string length too large");
        }
        std::string out;
        out.resize(static_cast<size_t>(length));
        if (length > 0) {
            read_bytes(out.data(), length);
        }
        return out;
    }

    std::string read_compact_string() {
        const uint32_t length = read_u32();
        std::string out;
        out.resize(static_cast<size_t>(length));
        if (length > 0) {
            read_bytes(out.data(), length);
        }
        return out;
    }

    pybind11::dict read_metadata(uint64_t kv_count, const std::string &duplicate_label) {
        pybind11::dict metadata;
        for (uint64_t i = 0; i < kv_count; i++) {
            std::string key = read_string();
            if (metadata.contains(pybind11::str(key))) {
                throw std::runtime_error(_filename + ": duplicate " + duplicate_label + " key: " + key);
            }
            metadata[pybind11::str(key)] = read_value();
        }
        return metadata;
    }

    pybind11::object read_value() {
        const uint32_t type = read_u32();
        return read_payload(type);
    }

    pybind11::object read_payload(uint32_t type) {
        switch (type) {
            case MEGA_META_UINT8: return pybind11::int_(read_le<uint8_t>());
            case MEGA_META_INT8: return pybind11::int_(read_le<int8_t>());
            case MEGA_META_UINT16: return pybind11::int_(read_le<uint16_t>());
            case MEGA_META_INT16: return pybind11::int_(read_le<int16_t>());
            case MEGA_META_UINT32: return pybind11::int_(read_le<uint32_t>());
            case MEGA_META_INT32: return pybind11::int_(read_le<int32_t>());
            case MEGA_META_UINT64: return pybind11::int_(read_le<uint64_t>());
            case MEGA_META_INT64: return pybind11::int_(read_le<int64_t>());
            case MEGA_META_FLOAT32: return pybind11::float_(read_le<float>());
            case MEGA_META_FLOAT64: return pybind11::float_(read_le<double>());
            case MEGA_META_BOOL: return pybind11::bool_(read_le<uint8_t>() != 0);
            case MEGA_META_STRING: return pybind11::str(read_string());
            case MEGA_META_ARRAY: {
                const uint32_t element_type = read_u32();
                const uint64_t length = read_u64();
                pybind11::list out;
                for (uint64_t i = 0; i < length; i++) {
                    out.append(read_payload(element_type));
                }
                return out;
            }
            default:
                throw std::runtime_error(_filename + ": unknown " + _format_name + " value type");
        }
    }
};

class mega_parser : public binary_metadata_parser {
public:
    mega_parser(const uint8_t *data, uint64_t size, const std::string &filename)
        : binary_metadata_parser(data, size, filename, "MEGA metadata") {}

    mega_parser(int fd, uint64_t size, const std::string &filename)
        : binary_metadata_parser(fd, size, filename, "MEGA metadata") {}

    pybind11::dict parse() {
        mega_footer_overlay footer = read_footer_overlay();
        const uint64_t parse_size = footer.present ? footer.base_size : _size;
        const uint64_t original_size = _size;
        _size = parse_size;

        const uint32_t magic = read_u32();
        if (magic != MEGA_MAGIC) {
            throw std::runtime_error(_filename + ": invalid MEGA magic");
        }
        const uint32_t version = read_u32();
        if (version != MEGA_FORMAT_VERSION) {
            throw std::runtime_error(_filename + ": unsupported MEGA format version; regenerate the artifact");
        }
        const uint64_t tensor_count = read_u64();
        const uint64_t kv_count = read_u64();

        pybind11::dict metadata = read_metadata(kv_count, "metadata");
        if (footer.present) {
            apply_footer_overlay(metadata, footer.metadata);
        }

        uint64_t chunk_count = 0;
        const pybind11::str chunk_count_key("mega.chunk_directory.count");
        if (metadata.contains(chunk_count_key)) {
            chunk_count = metadata[chunk_count_key].cast<uint64_t>();
            const pybind11::str chunk_format_key("mega.chunk_directory.format");
            if (!metadata.contains(chunk_format_key) ||
                metadata[chunk_format_key].cast<std::string>() != "v1") {
                throw std::runtime_error(_filename + ": chunk directory requires mega.chunk_directory.format=v1");
            }
        }
        uint64_t segment_count = 0;
        const pybind11::str segment_count_key("mega.segment_directory.count");
        if (metadata.contains(segment_count_key)) {
            segment_count = metadata[segment_count_key].cast<uint64_t>();
            const pybind11::str segment_format_key("mega.segment_directory.format");
            if (!metadata.contains(segment_format_key) ||
                metadata[segment_format_key].cast<std::string>() != "v1") {
                throw std::runtime_error(_filename + ": segment directory requires mega.segment_directory.format=v1");
            }
        }
        uint64_t moe_expert_count = 0;
        const pybind11::str moe_count_key("mega.moe.expert_table.count");
        if (metadata.contains(moe_count_key)) {
            moe_expert_count = metadata[moe_count_key].cast<uint64_t>();
            const pybind11::str moe_format_key("mega.moe.expert_table.format");
            if (!metadata.contains(moe_format_key) ||
                metadata[moe_format_key].cast<std::string>() != "v1") {
                throw std::runtime_error(_filename + ": MoE expert table requires mega.moe.expert_table.format=v1");
            }
        }
        uint64_t moe_n_experts = 0;
        const bool has_moe_n_experts = metadata.contains(pybind11::str("mega.moe.n_experts"));
        if (has_moe_n_experts) {
            moe_n_experts = metadata[pybind11::str("mega.moe.n_experts")].cast<uint64_t>();
        }

        std::set<std::string> tensor_names;
        std::vector<mega_tensor_dir_entry> tensor_dir;
        tensor_dir.reserve(static_cast<size_t>(tensor_count));
        std::vector<std::pair<uint64_t, uint64_t>> stored_ranges;
        for (uint64_t i = 0; i < tensor_count; i++) {
            std::string name = read_compact_string();
            if (!tensor_names.insert(name).second) {
                throw std::runtime_error(_filename + ": duplicate tensor name: " + name);
            }
            const uint32_t dir_flags = read_u32();
            constexpr uint32_t known_dir_flags =
                MEGA_TENSOR_DIR_HAS_STORAGE_FORMAT |
                MEGA_TENSOR_DIR_HAS_STORED_NBYTES |
                MEGA_TENSOR_DIR_HAS_TENSOR_FLAGS |
                MEGA_TENSOR_DIR_HAS_COMPRESSION_CODEC |
                MEGA_TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE |
                MEGA_TENSOR_DIR_HAS_CHECKSUM;
            if ((dir_flags & ~known_dir_flags) != 0) {
                throw std::runtime_error(_filename + ": tensor " + name + " has unknown directory flags");
            }
            const uint32_t n_dims = read_u32();
            std::vector<uint64_t> dims_reversed;
            dims_reversed.reserve(n_dims);
            for (uint32_t dim = 0; dim < n_dims; dim++) {
                dims_reversed.push_back(read_u64());
            }
            std::vector<uint64_t> shape_forward;
            shape_forward.reserve(dims_reversed.size());
            for (auto it = dims_reversed.rbegin(); it != dims_reversed.rend(); ++it) {
                shape_forward.push_back(*it);
            }
            std::string logical_dtype = read_compact_string();
            const uint64_t payload_offset = read_u64();
            const uint64_t logical_nbytes = infer_logical_nbytes(logical_dtype, shape_forward);
            std::string storage_format = "raw_dense";
            if ((dir_flags & MEGA_TENSOR_DIR_HAS_STORAGE_FORMAT) != 0) {
                storage_format = read_compact_string();
            }
            uint64_t stored_nbytes = logical_nbytes;
            if ((dir_flags & MEGA_TENSOR_DIR_HAS_STORED_NBYTES) != 0) {
                stored_nbytes = read_u64();
            }
            uint32_t tensor_flags = 0;
            if ((dir_flags & MEGA_TENSOR_DIR_HAS_TENSOR_FLAGS) != 0) {
                tensor_flags = read_u32();
            }
            uint32_t compression_codec = MEGA_COMPRESSION_NONE;
            if ((dir_flags & MEGA_TENSOR_DIR_HAS_COMPRESSION_CODEC) != 0) {
                compression_codec = read_u32();
            }
            uint32_t shuffle_elem_size = 0;
            if ((dir_flags & MEGA_TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE) != 0) {
                shuffle_elem_size = read_u32();
            }
            uint32_t checksum_type = MEGA_CHECKSUM_NONE;
            std::array<uint8_t, 32> checksum;
            checksum.fill(0);
            if ((dir_flags & MEGA_TENSOR_DIR_HAS_CHECKSUM) != 0) {
                checksum_type = read_u32();
                read_bytes(checksum.data(), checksum.size());
            }
            validate_checksum_field(checksum_type, checksum, _filename, "tensor " + name);
            if ((tensor_flags & ~uint32_t(0x3)) != 0) {
                throw std::runtime_error(_filename + ": tensor " + name + " has unknown tensor flags");
            }
            if (compression_codec > MEGA_COMPRESSION_ZSTD) {
                throw std::runtime_error(_filename + ": tensor " + name + " has unknown compression codec");
            }
            const bool compressed = (tensor_flags & uint32_t(0x1)) != 0;
            const bool byte_shuffled = (tensor_flags & uint32_t(0x2)) != 0;
            const bool raw_dense = storage_format == "raw_dense";
            const bool chunked_raw_dense = storage_format == "chunked_raw_dense";
            if (logical_dtype.empty() || storage_format.empty()) {
                throw std::runtime_error(_filename + ": tensor " + name + " has empty dtype or storage format");
            }
            if (!compressed && compression_codec != 0) {
                throw std::runtime_error(_filename + ": uncompressed tensor " + name + " has non-NONE codec");
            }
            if (raw_dense && !compressed && stored_nbytes != logical_nbytes) {
                throw std::runtime_error(_filename + ": raw_dense tensor " + name + " stored_nbytes != logical_nbytes");
            }
            if (compressed && compression_codec == 0) {
                throw std::runtime_error(_filename + ": compressed tensor " + name + " has NONE codec");
            }
            if (byte_shuffled && shuffle_elem_size == 0) {
                throw std::runtime_error(_filename + ": byte-shuffled tensor " + name + " has zero shuffle size");
            }
            if (byte_shuffled && logical_nbytes % shuffle_elem_size != 0) {
                throw std::runtime_error(_filename + ": byte-shuffled tensor " + name + " has incompatible logical size");
            }

            if (payload_offset > UINT64_MAX - stored_nbytes) {
                throw std::runtime_error(_filename + ": tensor " + name + " payload range overflows");
            }
            if (!chunked_raw_dense && stored_nbytes > 0) {
                stored_ranges.emplace_back(payload_offset, payload_offset + stored_nbytes);
            }

            tensor_dir.push_back(mega_tensor_dir_entry{
                name,
                logical_dtype,
                storage_format,
                shape_forward,
                payload_offset,
                logical_nbytes,
                stored_nbytes,
                tensor_flags,
                compression_codec,
                shuffle_elem_size,
                checksum_type,
                checksum,
            });
        }

        std::vector<std::vector<mega_chunk_record>> chunks_by_tensor(static_cast<size_t>(tensor_count));
        std::set<std::pair<uint32_t, uint32_t>> seen_chunks;
        for (uint64_t i = 0; i < chunk_count; i++) {
            mega_chunk_record rec;
            rec.tensor_id = read_u32();
            rec.chunk_id = read_u32();
            rec.logical_offset = read_u64();
            rec.logical_size = read_u64();
            rec.payload_offset = read_u64();
            rec.stored_size = read_u64();
            rec.codec = read_u32();
            rec.flags = read_u32();
            rec.checksum_type = read_u32();
            const uint32_t reserved = read_u32();
            rec.checksum.fill(0);
            read_bytes(rec.checksum.data(), rec.checksum.size());

            if (rec.tensor_id >= tensor_count) {
                throw std::runtime_error(_filename + ": chunk record references invalid tensor_id");
            }
            if (!seen_chunks.insert({rec.tensor_id, rec.chunk_id}).second) {
                throw std::runtime_error(_filename + ": duplicate chunk record");
            }
            if (reserved != 0) {
                throw std::runtime_error(_filename + ": chunk record has non-zero reserved field");
            }
            if ((rec.flags & ~(MEGA_TENSOR_FLAG_COMPRESSED | MEGA_TENSOR_FLAG_BYTE_SHUFFLED)) != 0) {
                throw std::runtime_error(_filename + ": chunk record has unknown flags");
            }
            if (rec.codec > MEGA_COMPRESSION_ZSTD) {
                throw std::runtime_error(_filename + ": chunk record has unknown codec");
            }
            const bool chunk_compressed = (rec.flags & MEGA_TENSOR_FLAG_COMPRESSED) != 0;
            const bool chunk_byte_shuffled = (rec.flags & MEGA_TENSOR_FLAG_BYTE_SHUFFLED) != 0;
            if (!chunk_compressed && rec.codec != MEGA_COMPRESSION_NONE) {
                throw std::runtime_error(_filename + ": uncompressed chunk has non-NONE codec");
            }
            if (chunk_compressed && rec.codec == MEGA_COMPRESSION_NONE) {
                throw std::runtime_error(_filename + ": compressed chunk has NONE codec");
            }
            if (rec.logical_size == 0 || rec.stored_size == 0) {
                throw std::runtime_error(_filename + ": chunk record has zero size");
            }
            const auto &td = tensor_dir[rec.tensor_id];
            if (td.storage_format != "chunked_raw_dense") {
                throw std::runtime_error(_filename + ": chunk record attached to non-chunked tensor");
            }
            if (rec.logical_offset > td.logical_nbytes ||
                rec.logical_size > td.logical_nbytes - rec.logical_offset) {
                throw std::runtime_error(_filename + ": chunk logical range exceeds tensor logical size");
            }
            if (chunk_byte_shuffled && (td.shuffle_elem_size == 0 || rec.logical_size % td.shuffle_elem_size != 0)) {
                throw std::runtime_error(_filename + ": byte-shuffled chunk has incompatible shuffle size");
            }
            if (rec.payload_offset > UINT64_MAX - rec.stored_size) {
                throw std::runtime_error(_filename + ": chunk payload range overflows");
            }
            validate_checksum_field(rec.checksum_type, rec.checksum, _filename, "chunk record");

            chunks_by_tensor[rec.tensor_id].push_back(rec);
            stored_ranges.emplace_back(rec.payload_offset, rec.payload_offset + rec.stored_size);
        }

        for (size_t tensor_id = 0; tensor_id < tensor_dir.size(); tensor_id++) {
            auto &td = tensor_dir[tensor_id];
            auto &chunks = chunks_by_tensor[tensor_id];
            if (td.storage_format == "chunked_raw_dense") {
                if (td.logical_nbytes > 0 && chunks.empty()) {
                    throw std::runtime_error(_filename + ": chunked_raw_dense tensor has no chunks");
                }
                std::sort(chunks.begin(), chunks.end(), [](const mega_chunk_record &a, const mega_chunk_record &b) {
                    if (a.logical_offset != b.logical_offset) {
                        return a.logical_offset < b.logical_offset;
                    }
                    return a.chunk_id < b.chunk_id;
                });
                validate_chunk_records(chunks, td.logical_nbytes, td.shuffle_elem_size, _filename, td.name);
            } else if (!chunks.empty()) {
                throw std::runtime_error(_filename + ": non-chunked tensor has chunk records");
            }

        }

        pybind11::list segments;
        std::set<uint32_t> seen_segments;
        std::vector<std::pair<uint64_t, uint64_t>> segment_ranges;
        for (uint64_t i = 0; i < segment_count; i++) {
            mega_segment_record rec;
            rec.segment_id = read_u32();
            rec.kind = read_u32();
            rec.priority = read_u32();
            rec.flags = read_u32();
            rec.layer_start = read_u32();
            rec.layer_end = read_u32();
            rec.expert_start = read_u32();
            rec.expert_end = read_u32();
            rec.payload_offset = read_u64();
            rec.payload_size = read_u64();
            rec.first_tensor_id = read_u32();
            rec.tensor_count = read_u32();
            rec.prefetch_group = read_u32();
            rec.device_hint = read_u32();
            rec.cache_policy = read_u32();
            const uint32_t reserved = read_u32();

            if (!seen_segments.insert(rec.segment_id).second) {
                throw std::runtime_error(_filename + ": duplicate segment_id");
            }
            if (reserved != 0) {
                throw std::runtime_error(_filename + ": segment record has non-zero reserved field");
            }
            if (rec.layer_end < rec.layer_start) {
                throw std::runtime_error(_filename + ": segment layer range is reversed");
            }
            if (rec.expert_end < rec.expert_start) {
                throw std::runtime_error(_filename + ": segment expert range is reversed");
            }
            if (rec.first_tensor_id > tensor_count ||
                rec.tensor_count > tensor_count - rec.first_tensor_id) {
                throw std::runtime_error(_filename + ": segment tensor range exceeds tensor directory");
            }
            if (rec.payload_offset > UINT64_MAX - rec.payload_size) {
                throw std::runtime_error(_filename + ": segment payload range overflows");
            }
            if (rec.payload_size > 0) {
                segment_ranges.emplace_back(rec.payload_offset, rec.payload_offset + rec.payload_size);
            }

            pybind11::dict py_segment;
            py_segment["segment_id"] = rec.segment_id;
            py_segment["kind"] = rec.kind;
            py_segment["priority"] = rec.priority;
            py_segment["flags"] = rec.flags;
            py_segment["layer_start"] = rec.layer_start;
            py_segment["layer_end"] = rec.layer_end;
            py_segment["expert_start"] = rec.expert_start;
            py_segment["expert_end"] = rec.expert_end;
            py_segment["payload_offset"] = rec.payload_offset;
            py_segment["payload_size"] = rec.payload_size;
            py_segment["first_tensor_id"] = rec.first_tensor_id;
            py_segment["tensor_count"] = rec.tensor_count;
            py_segment["prefetch_group"] = rec.prefetch_group;
            py_segment["device_hint"] = rec.device_hint;
            py_segment["cache_policy"] = rec.cache_policy;
            segments.append(py_segment);
        }

        pybind11::list moe_experts;
        std::set<uint32_t> seen_moe_tensors;
        for (uint64_t i = 0; i < moe_expert_count; i++) {
            mega_moe_expert_record rec;
            rec.tensor_id = read_u32();
            rec.layer_id = read_u32();
            rec.expert_id = read_u32();
            rec.expert_role = read_u32();
            rec.segment_id = read_u32();
            rec.prefetch_group = read_u32();
            rec.cache_policy = read_u32();
            const uint32_t reserved = read_u32();

            if (reserved != 0) {
                throw std::runtime_error(_filename + ": MoE expert record has non-zero reserved field");
            }
            if (rec.tensor_id >= tensor_count) {
                throw std::runtime_error(_filename + ": MoE expert record references invalid tensor_id");
            }
            if (!seen_moe_tensors.insert(rec.tensor_id).second) {
                throw std::runtime_error(_filename + ": duplicate MoE expert tensor_id");
            }
            if (has_moe_n_experts && rec.expert_id >= moe_n_experts) {
                throw std::runtime_error(_filename + ": MoE expert_id exceeds mega.moe.n_experts");
            }
            if (rec.segment_id != MEGA_NO_SEGMENT_ID &&
                !seen_segments.empty() &&
                seen_segments.find(rec.segment_id) == seen_segments.end()) {
                throw std::runtime_error(_filename + ": MoE expert record references unknown segment_id");
            }

            pybind11::dict py_expert;
            py_expert["tensor_id"] = rec.tensor_id;
            py_expert["tensor_name"] = tensor_dir[rec.tensor_id].name;
            py_expert["layer_id"] = rec.layer_id;
            py_expert["expert_id"] = rec.expert_id;
            py_expert["expert_role"] = rec.expert_role;
            py_expert["segment_id"] = rec.segment_id;
            py_expert["prefetch_group"] = rec.prefetch_group;
            py_expert["cache_policy"] = rec.cache_policy;
            moe_experts.append(py_expert);
        }

        uint64_t alignment = MEGA_DEFAULT_ALIGNMENT;
        const pybind11::str alignment_key("general.alignment");
        if (metadata.contains(alignment_key)) {
            alignment = metadata[alignment_key].cast<uint64_t>();
            if (alignment == 0) {
                throw std::runtime_error(_filename + ": general.alignment must be non-zero");
            }
        }
        const uint64_t header_length = align_up(_pos, alignment, "MEGA");
        if (header_length > _size) {
            throw std::runtime_error(_filename + ": metadata extends beyond file size");
        }
        const uint64_t data_size = _size - header_length;
        std::sort(stored_ranges.begin(), stored_ranges.end());
        uint64_t prev_end = 0;
        for (const auto &range : stored_ranges) {
            if (range.first < prev_end) {
                throw std::runtime_error(_filename + ": tensor payload ranges overlap");
            }
            if (range.second > data_size) {
                throw std::runtime_error(_filename + ": tensor payload range exceeds file size");
            }
            prev_end = range.second;
        }
        for (const auto &range : segment_ranges) {
            if (range.second > data_size) {
                throw std::runtime_error(_filename + ": segment payload range exceeds file size");
            }
        }

        pybind11::dict out;
        out["version"] = version;
        out["metadata"] = metadata;
        pybind11::list tensor_records;
        for (size_t tensor_id = 0; tensor_id < tensor_dir.size(); tensor_id++) {
            const auto &td = tensor_dir[tensor_id];
            pybind11::list py_shape;
            for (uint64_t dim : td.shape) {
                py_shape.append(dim);
            }
            pybind11::list py_chunks;
            for (const auto &chunk : chunks_by_tensor[tensor_id]) {
                py_chunks.append(pybind11::make_tuple(
                    chunk.tensor_id,
                    chunk.chunk_id,
                    chunk.logical_offset,
                    chunk.logical_size,
                    chunk.payload_offset,
                    chunk.stored_size,
                    chunk.codec,
                    chunk.flags,
                    chunk.checksum_type,
                    pybind11::bytes(
                        reinterpret_cast<const char *>(chunk.checksum.data()),
                        chunk.checksum.size()
                    )
                ));
            }
            tensor_records.append(pybind11::make_tuple(
                td.name,
                tensor_id,
                td.logical_dtype,
                py_shape,
                td.payload_offset,
                td.logical_nbytes,
                td.stored_nbytes,
                td.tensor_flags,
                td.compression_codec,
                td.shuffle_elem_size,
                td.checksum_type,
                pybind11::bytes(
                    reinterpret_cast<const char *>(td.checksum.data()),
                    td.checksum.size()
                ),
                td.storage_format,
                py_chunks
            ));
        }
        out["tensor_records"] = tensor_records;
        out["segments"] = segments;
        out["moe_experts"] = moe_experts;
        out["header_length"] = header_length;
        out["payload_end"] = parse_size;
        pybind11::dict footer_out;
        footer_out["present"] = footer.present;
        footer_out["generation"] = footer.generation;
        footer_out["overlay_offset"] = footer.overlay_offset;
        footer_out["overlay_size"] = footer.overlay_size;
        footer_out["base_size"] = footer.base_size;
        if (footer.present) {
            footer_out["checksum"] = pybind11::bytes(
                reinterpret_cast<const char *>(footer.checksum.data()),
                footer.checksum.size()
            );
        } else {
            footer_out["checksum"] = pybind11::bytes("", 0);
        }
        out["footer_overlay"] = footer_out;
        _size = original_size;
        return out;
    }

private:
    mega_footer_overlay read_footer_overlay() {
        mega_footer_overlay footer;
        if (_size < MEGA_FOOTER_TRAILER_SIZE) {
            return footer;
        }
        const uint64_t trailer = _size - MEGA_FOOTER_TRAILER_SIZE;
        const uint32_t magic = read_le_at<uint32_t>(trailer);
        if (magic != MEGA_FOOTER_MAGIC) {
            return footer;
        }
        footer.generation = read_le_at<uint64_t>(trailer + 4);
        footer.overlay_offset = read_le_at<uint64_t>(trailer + 12);
        footer.overlay_size = read_le_at<uint64_t>(trailer + 20);
        footer.base_size = read_le_at<uint64_t>(trailer + 28);
        read_bytes_at(trailer + 36, footer.checksum.data(), footer.checksum.size());

        if (footer.generation == 0) {
            throw std::runtime_error(_filename + ": MEGA footer overlay generation must be non-zero");
        }
        if (footer.base_size == 0 || footer.base_size > trailer) {
            throw std::runtime_error(_filename + ": MEGA footer overlay base size is invalid");
        }
        if (footer.overlay_size == 0) {
            throw std::runtime_error(_filename + ": MEGA footer overlay is empty");
        }
        if (footer.overlay_offset < footer.base_size ||
            footer.overlay_offset > trailer ||
            footer.overlay_size > trailer - footer.overlay_offset) {
            throw std::runtime_error(_filename + ": MEGA footer overlay range is invalid");
        }
        if (footer.overlay_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error(_filename + ": MEGA footer overlay is too large for this platform");
        }

        std::vector<uint8_t> overlay(static_cast<size_t>(footer.overlay_size));
        read_bytes_at(footer.overlay_offset, overlay.data(), footer.overlay_size);
        const auto actual = sha256_bytes(overlay);
        if (!std::equal(actual.begin(), actual.end(), footer.checksum.begin())) {
            throw std::runtime_error(_filename + ": MEGA footer overlay checksum mismatch");
        }

        binary_metadata_parser overlay_parser(
            overlay.data(),
            footer.overlay_size,
            _filename,
            "MEGA footer overlay"
        );
        footer.metadata = overlay_parser.parse_footer_metadata();
        footer.present = true;
        return footer;
    }

    static void apply_footer_overlay(pybind11::dict &metadata, const pybind11::dict &overlay) {
        static const std::set<std::string> reserved_keys = {
            "general.alignment",
            "mega.tensor_info.format",
            "mega.chunk_directory.format",
            "mega.chunk_directory.count",
            "mega.segment_directory.format",
            "mega.segment_directory.count",
            "mega.moe.expert_table.format",
            "mega.moe.expert_table.count",
        };
        for (auto item : overlay) {
            std::string key = pybind11::cast<std::string>(item.first);
            if (reserved_keys.find(key) != reserved_keys.end()) {
                throw std::runtime_error("MEGA footer overlay cannot override structural key: " + key);
            }
            metadata[pybind11::str(key)] = item.second;
        }
    }
};

pybind11::dict parse_metadata_buffer(uintptr_t ptr, uint64_t size, const std::string &filename) {
    mega_parser parser(reinterpret_cast<const uint8_t *>(ptr), size, filename);
    return parser.parse();
}

pybind11::dict parse_metadata_fd(int fd, const std::string &filename, uint64_t size) {
    mega_parser parser(fd, size, filename);
    return parser.parse();
}

static void append_footer_overlay_file(
    const std::string &filename,
    pybind11::dict metadata,
    uint64_t generation
) {
    if (generation == 0) {
        throw std::runtime_error("MEGA footer overlay generation must be non-zero");
    }
    std::vector<uint8_t> overlay;
    append_le<uint32_t>(overlay, MEGA_FOOTER_MAGIC);
    append_le<uint64_t>(overlay, static_cast<uint64_t>(metadata.size()));
    append_metadata_entries(overlay, metadata);
    const auto checksum = sha256_bytes(overlay);

    const int fd = open(filename.c_str(), O_RDWR
#ifdef _MSC_VER
        | _O_BINARY
#endif
        , 0644);
    if (fd < 0) {
        throw std::runtime_error(filename + ": failed to open MEGA file for footer overlay");
    }
    try {
        const int64_t physical_size_signed = lseek(fd, 0, SEEK_END);
        if (physical_size_signed < 0) {
            throw std::runtime_error(filename + ": failed to seek MEGA file");
        }
        const uint64_t physical_size = static_cast<uint64_t>(physical_size_signed);
        pybind11::dict parsed = parse_metadata_fd(fd, filename, physical_size);
        const uint64_t base_size = parsed["payload_end"].cast<uint64_t>();
        const uint64_t overlay_offset = physical_size;

        std::vector<uint8_t> trailer;
        append_le<uint32_t>(trailer, MEGA_FOOTER_MAGIC);
        append_le<uint64_t>(trailer, generation);
        append_le<uint64_t>(trailer, overlay_offset);
        append_le<uint64_t>(trailer, static_cast<uint64_t>(overlay.size()));
        append_le<uint64_t>(trailer, base_size);
        append_bytes(trailer, checksum.data(), checksum.size());
        if (trailer.size() != MEGA_FOOTER_TRAILER_SIZE) {
            throw std::runtime_error("MEGA footer trailer size mismatch");
        }

        pybind11::gil_scoped_release release;
        if (lseek(fd, 0, SEEK_END) < 0) {
            throw std::runtime_error(filename + ": failed to seek before footer write");
        }
        write_all_fd(fd, overlay, filename);
        write_all_fd(fd, trailer, filename);
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
}

class megakv_parser : public binary_metadata_parser {
public:
    megakv_parser(int fd, uint64_t size, const std::string &filename)
        : binary_metadata_parser(fd, size, filename, "MEGAKV") {}

    pybind11::dict parse() {
        const uint32_t magic = read_u32();
        if (magic != MEGAKV_MAGIC) {
            throw std::runtime_error(_filename + ": invalid MEGAKV magic");
        }
        const uint32_t version = read_u32();
        if (version != 1) {
            throw std::runtime_error(_filename + ": unsupported MEGAKV version");
        }
        const uint64_t entry_count = read_u64();
        const uint64_t kv_count = read_u64();

        pybind11::dict metadata = read_metadata(kv_count, "MEGAKV metadata");

        uint64_t alignment = MEGA_DEFAULT_ALIGNMENT;
        const pybind11::str alignment_key("general.alignment");
        if (metadata.contains(alignment_key)) {
            alignment = metadata[alignment_key].cast<uint64_t>();
            if (alignment == 0) {
                throw std::runtime_error(_filename + ": general.alignment must be non-zero");
            }
        }

        pybind11::list entries;
        std::vector<std::pair<uint64_t, uint64_t>> stored_ranges;
        for (uint64_t i = 0; i < entry_count; ++i) {
            megakv_entry_record rec;
            rec.name = read_string();
            rec.layer_id = read_u32();
            rec.kv_role = read_u32();
            rec.sequence_id = read_u64();
            rec.token_start = read_u64();
            rec.token_count = read_u64();
            const uint32_t n_dims = read_u32();
            rec.shape.reserve(n_dims);
            for (uint32_t dim = 0; dim < n_dims; ++dim) {
                rec.shape.push_back(read_u64());
            }
            rec.logical_dtype = read_string();
            rec.payload_offset = read_u64();
            rec.logical_nbytes = read_u64();
            rec.stored_nbytes = read_u64();
            rec.flags = read_u32();
            rec.codec = read_u32();
            rec.shuffle_elem_size = read_u32();
            const uint32_t reserved = read_u32();
            rec.checksum_type = read_u32();
            rec.checksum.fill(0);
            read_bytes(rec.checksum.data(), rec.checksum.size());

            if (rec.name.empty() || rec.logical_dtype.empty()) {
                throw std::runtime_error(_filename + ": MEGAKV entry has empty name or dtype");
            }
            if (reserved != 0) {
                throw std::runtime_error(_filename + ": MEGAKV entry has non-zero reserved field");
            }
            if ((rec.flags & ~(MEGA_TENSOR_FLAG_COMPRESSED | MEGA_TENSOR_FLAG_BYTE_SHUFFLED)) != 0) {
                throw std::runtime_error(_filename + ": MEGAKV entry has unknown flags");
            }
            if (rec.codec > MEGA_COMPRESSION_ZSTD) {
                throw std::runtime_error(_filename + ": MEGAKV entry has unknown codec");
            }
            const bool compressed = (rec.flags & MEGA_TENSOR_FLAG_COMPRESSED) != 0;
            const bool byte_shuffled = (rec.flags & MEGA_TENSOR_FLAG_BYTE_SHUFFLED) != 0;
            if (!compressed && rec.codec != MEGA_COMPRESSION_NONE) {
                throw std::runtime_error(_filename + ": uncompressed MEGAKV entry has non-NONE codec");
            }
            if (compressed && rec.codec == MEGA_COMPRESSION_NONE) {
                throw std::runtime_error(_filename + ": compressed MEGAKV entry has NONE codec");
            }
            if (byte_shuffled && (rec.shuffle_elem_size == 0 || rec.logical_nbytes % rec.shuffle_elem_size != 0)) {
                throw std::runtime_error(_filename + ": byte-shuffled MEGAKV entry has incompatible shuffle size");
            }
            if (rec.payload_offset > UINT64_MAX - rec.stored_nbytes) {
                throw std::runtime_error(_filename + ": MEGAKV entry payload range overflows");
            }
            validate_checksum_field(rec.checksum_type, rec.checksum, _filename, "MEGAKV entry");
            if (rec.stored_nbytes > 0) {
                stored_ranges.emplace_back(rec.payload_offset, rec.payload_offset + rec.stored_nbytes);
            }

            pybind11::list shape;
            for (uint64_t dim : rec.shape) {
                shape.append(dim);
            }
            pybind11::dict entry;
            entry["entry_id"] = i;
            entry["name"] = pybind11::str(rec.name);
            entry["layer_id"] = rec.layer_id;
            entry["kv_role"] = rec.kv_role;
            entry["sequence_id"] = rec.sequence_id;
            entry["token_start"] = rec.token_start;
            entry["token_count"] = rec.token_count;
            entry["shape"] = shape;
            entry["dtype"] = pybind11::str(rec.logical_dtype);
            entry["payload_offset"] = rec.payload_offset;
            entry["logical_nbytes"] = rec.logical_nbytes;
            entry["stored_nbytes"] = rec.stored_nbytes;
            entry["flags"] = rec.flags;
            entry["codec"] = rec.codec;
            entry["shuffle_elem_size"] = rec.shuffle_elem_size;
            entry["checksum_type"] = rec.checksum_type;
            entry["checksum"] = pybind11::bytes(
                reinterpret_cast<const char *>(rec.checksum.data()),
                rec.checksum.size()
            );
            entries.append(entry);
        }

        const uint64_t payload_offset = align_up(_pos, alignment, "MEGAKV");
        if (payload_offset > _size) {
            throw std::runtime_error(_filename + ": MEGAKV directory extends beyond file size");
        }
        const uint64_t payload_size = _size - payload_offset;
        std::sort(stored_ranges.begin(), stored_ranges.end());
        uint64_t prev_end = 0;
        for (const auto &range : stored_ranges) {
            if (range.first < prev_end) {
                throw std::runtime_error(_filename + ": MEGAKV entry payload ranges overlap");
            }
            if (range.second > payload_size) {
                throw std::runtime_error(_filename + ": MEGAKV entry payload range exceeds file size");
            }
            prev_end = range.second;
        }

        pybind11::dict out;
        out["version"] = version;
        out["metadata"] = metadata;
        out["entries"] = entries;
        out["payload_offset"] = payload_offset;
        return out;
    }
};

pybind11::dict parse_kv_fd(int fd, const std::string &filename, uint64_t size) {
    megakv_parser parser(fd, size, filename);
    return parser.parse();
}

uint32_t payload_crc32_fd(int fd, const std::string &filename, uint64_t header_length, uint64_t size) {
    if (header_length > size) {
        throw std::runtime_error(filename + ": header length exceeds file size");
    }
    return crc32_fd_range(fd, filename, header_length, size - header_length);
}

} // namespace

// Bindings

// Async host-to-device memcpy for unified memory copier
static int memcpy_h2d_async(uintptr_t dst, uintptr_t src, size_t size) {
    if (!cuda_fns.cudaMemcpyAsync) {
        return -1;
    }
    cudaError_t err = cuda_fns.cudaMemcpyAsync(
        reinterpret_cast<void *>(dst),
        reinterpret_cast<const void *>(src),
        size,
        cudaMemcpyHostToDevice,
        nullptr  // default stream
    );
    return static_cast<int>(err);
}

PYBIND11_MODULE(__MOD_NAME__, m)
{
#ifdef _MSC_VER
    init_dstorage_bindings(m);
#endif
    // Initialize GIL release setting from environment variable on module load
    init_gil_release_from_env();
    m.def("is_cuda_found", &is_cuda_found);
    m.def("is_hip_found", &is_hip_found);
    m.def("is_cufile_found", &is_cufile_found);
    m.def("cufile_version", &cufile_version);
    m.def("set_debug_log", &set_debug_log);
    m.def("get_alignment_size", &get_alignment_size);
    m.def("is_gds_supported", &is_gds_supported);
    m.def("init_gds", &init_gds);
    m.def("close_gds", &close_gds);
    m.def("get_device_pci_bus", &get_device_pci_bus);
    m.def("set_numa_node", &set_numa_node);
    m.def("read_buffer", &read_buffer);
    m.def("parse_metadata_fd", &parse_metadata_fd);
    m.def("parse_metadata_buffer", &parse_metadata_buffer);
    m.def("parse_kv_fd", &parse_kv_fd);
    m.def("encode_file_tensors", &encode_file_tensors);
    m.def("write_file", &write_file);
    m.def("write_kv_file", &write_kv_file);
    m.def("append_footer_overlay_file", &append_footer_overlay_file);
    m.def("payload_crc32_fd", &payload_crc32_fd);
    m.def(
        "payload_sha256_fd",
        [](int fd, const std::string &filename, uint64_t header_length, uint64_t size) {
            std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
            {
                pybind11::gil_scoped_release release;
                if (header_length > size) {
                    throw std::runtime_error(filename + ": header length exceeds file size");
                }
                digest = sha256_fd_range(fd, filename, header_length, size - header_length);
            }
            return pybind11::bytes(reinterpret_cast<const char *>(digest.data()), digest.size());
        }
    );
    m.def(
        "verify_x509_signature",
        [](const std::string &leaf_pem,
           const std::string &chain_pem,
           const std::string &trusted_roots_pem,
           pybind11::bytes py_statement,
           pybind11::bytes py_signature,
           const std::string &algorithm) {
            const std::string statement = py_statement;
            const std::string signature = py_signature;
            return verify_x509_signature(
                leaf_pem,
                chain_pem,
                trusted_roots_pem,
                statement,
                signature,
                algorithm
            );
        }
    );
    m.def(
        "decode_payload_fd",
        [](int fd,
           const std::string &filename,
           const std::string &tensor_name,
           uint64_t file_offset,
           uint64_t stored_nbytes,
           uint64_t logical_nbytes,
           uint32_t tensor_flags,
           uint32_t compression_codec,
           uint32_t shuffle_elem_size,
           uint32_t checksum_type,
           pybind11::bytes py_checksum,
           uintptr_t dst_ptr,
           bool dst_is_cuda) {
            const std::string checksum_bytes = py_checksum;
            if (checksum_bytes.size() != 32) {
                throw std::runtime_error("MEGA tensor checksum field must be 32 bytes");
            }
            std::array<uint8_t, 32> checksum;
            std::memcpy(checksum.data(), checksum_bytes.data(), checksum.size());
            pybind11::gil_scoped_release release;
            decode_payload_fd(
                fd,
                filename,
                tensor_name,
                file_offset,
                stored_nbytes,
                logical_nbytes,
                tensor_flags,
                compression_codec,
                shuffle_elem_size,
                checksum_type,
                checksum,
                dst_ptr,
                dst_is_cuda
            );
        }
    );
    m.def(
        "decode_kv_entry_fd",
        [](int fd,
           const std::string &filename,
           const std::string &entry_name,
           uint64_t file_offset,
           uint64_t stored_nbytes,
           uint64_t logical_nbytes,
           uint32_t flags,
           uint32_t codec,
           uint32_t shuffle_elem_size,
           uint32_t checksum_type,
           pybind11::bytes py_checksum,
           uintptr_t dst_ptr,
           bool dst_is_cuda) {
            const std::string checksum_bytes = py_checksum;
            if (checksum_bytes.size() != 32) {
                throw std::runtime_error("MEGAKV checksum field must be 32 bytes");
            }
            std::array<uint8_t, 32> checksum;
            std::memcpy(checksum.data(), checksum_bytes.data(), checksum.size());
            pybind11::gil_scoped_release release;
            std::vector<uint8_t> payload = read_decode_payload(
                fd,
                filename,
                entry_name,
                file_offset,
                stored_nbytes,
                logical_nbytes,
                flags,
                codec,
                shuffle_elem_size,
                checksum_type,
                checksum,
                "MEGAKV entry"
            );
            copy_decoded_payload_to_dst(dst_ptr, dst_is_cuda, payload);
        }
    );
    m.def(
        "decode_chunks_fd",
        [](int fd,
           const std::string &filename,
           const std::string &tensor_name,
           uint64_t header_length,
           pybind11::list py_chunks,
           uint64_t logical_nbytes,
           uint32_t shuffle_elem_size,
           uintptr_t dst_ptr,
           bool dst_is_cuda) {
            std::vector<mega_chunk_record> chunks;
            chunks.reserve(static_cast<size_t>(pybind11::len(py_chunks)));
            for (pybind11::handle item : py_chunks) {
                pybind11::dict d = pybind11::reinterpret_borrow<pybind11::dict>(item);
                mega_chunk_record rec;
                rec.tensor_id = d["tensor_id"].cast<uint32_t>();
                rec.chunk_id = d["chunk_id"].cast<uint32_t>();
                rec.logical_offset = d["logical_offset"].cast<uint64_t>();
                rec.logical_size = d["logical_size"].cast<uint64_t>();
                rec.payload_offset = d["payload_offset"].cast<uint64_t>();
                rec.stored_size = d["stored_size"].cast<uint64_t>();
                rec.codec = d["codec"].cast<uint32_t>();
                rec.flags = d["flags"].cast<uint32_t>();
                rec.checksum_type = d["checksum_type"].cast<uint32_t>();
                rec.checksum.fill(0);
                if (d.contains("checksum")) {
                    const std::string checksum = d["checksum"].cast<std::string>();
                    if (checksum.size() != rec.checksum.size()) {
                        throw std::runtime_error("MEGA chunk checksum field must be 32 bytes");
                    }
                    std::memcpy(rec.checksum.data(), checksum.data(), rec.checksum.size());
                }
                chunks.push_back(rec);
            }
            pybind11::gil_scoped_release release;
            decode_chunks_fd(
                fd,
                filename,
                tensor_name,
                header_length,
                chunks,
                logical_nbytes,
                shuffle_elem_size,
                dst_ptr,
                dst_is_cuda
            );
        }
    );
    m.def("cpu_malloc", &cpu_malloc);
    m.def("cpu_free", &cpu_free);
    m.def("gpu_malloc", &gpu_malloc);
    m.def("gpu_free", &gpu_free);
    m.def("load_library_functions", &load_library_functions,
          pybind11::arg("cudart_lib_name") = "");
    m.def("memcpy_h2d_async", &memcpy_h2d_async);
    m.def("get_cpp_metrics", &get_cpp_metrics);
    m.def("set_gil_release", &set_gil_release);
    m.def("get_gil_release", &get_gil_release);

    pybind11::class_<gds_device_buffer>(m, "gds_device_buffer")
        .def(pybind11::init<const uintptr_t, const uint64_t, bool>())
        .def("cufile_register", &gds_device_buffer::cufile_register)
        .def("cufile_deregister", &gds_device_buffer::cufile_deregister)
        .def("memmove", &gds_device_buffer::memmove)
        .def("get_base_address", &gds_device_buffer::get_base_address)
        .def("get_length", &gds_device_buffer::get_length);

    // Helper lambdas to conditionally apply GIL release
    auto nogds_submit_read = [](nogds_file_reader& self, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fd, dst, offset, length, ptr_off);
        } else {
            return self.submit_read(fd, dst, offset, length, ptr_off);
        }
    };

    auto nogds_wait_read = [](nogds_file_reader& self, const int thread_id) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.wait_read(thread_id);
        } else {
            return self.wait_read(thread_id);
        }
    };

    pybind11::class_<nogds_file_reader>(m, "nogds_file_reader")
        .def(pybind11::init<const bool, const uint64_t, const int, bool, int>())
        .def("submit_read", nogds_submit_read)
        .def("wait_read", nogds_wait_read);

    pybind11::class_<gds_file_handle>(m, "gds_file_handle")
        .def(pybind11::init<std::string, bool, bool>());

    // Helper lambdas for gds_file_reader to conditionally apply GIL release
    auto gds_submit_read = [](gds_file_reader& self, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fh, dst, offset, length, ptr_off, file_length);
        } else {
            return self.submit_read(fh, dst, offset, length, ptr_off, file_length);
        }
    };

    auto gds_wait_read = [](gds_file_reader& self, const int id) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.wait_read(id);
        } else {
            return self.wait_read(id);
        }
    };

    pybind11::class_<gds_file_reader>(m, "gds_file_reader")
        .def(pybind11::init<const int, bool, int>())
        .def("submit_read", gds_submit_read)
        .def("wait_read", gds_wait_read);

    pybind11::class_<cpp_metrics_t>(m, "cpp_metrics")
        .def(pybind11::init<>())
        .def_readwrite("bounce_buffer_bytes", &cpp_metrics_t::bounce_buffer_bytes);
}
