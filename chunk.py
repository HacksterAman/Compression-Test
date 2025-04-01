import os
import time
import numpy as np
import sys
import platform
import psutil
import traceback
import cupy as cp
import pyzstd
from nvidia import nvcomp

def get_system_info():
    info = {
        "Platform": platform.platform(),
        "Python": sys.version.split()[0],
        "Processor": platform.processor(),
        "Memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
    }
    print("System Info:", *[f"  {k}: {v}" for k, v in info.items()], sep="\n")

def generate_test_data(filename, size_bytes):
    try:
        chunk_size = 1024
        pattern = bytes([i % 256 for i in range(chunk_size)])
        with open(filename, 'wb') as f:
            for _ in range(size_bytes // chunk_size):
                f.write(pattern)
            f.write(pattern[:size_bytes % chunk_size])
        return True
    except Exception as e:
        print(f"Error generating {filename}: {e}")
        return False

def cpu_compression(test_file, chunk_size=4 * 1024 * 1024):
    compressor, decompressor = pyzstd.ZstdCompressor(), pyzstd.ZstdDecompressor()
    total_size, total_compressed = os.path.getsize(test_file), 0
    start, ok = time.time(), True
    with open(test_file, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            compressed = compressor.compress(chunk)
            ok &= chunk == decompressor.decompress(compressed)
            total_compressed += len(compressed)
    return ok, total_size / total_compressed if total_compressed else 0, time.time() - start

def gpu_compression(test_file, chunk_size=4 * 1024 * 1024):
    codec, total_size, total_compressed = nvcomp.Codec(algorithm="Zstd"), os.path.getsize(test_file), 0
    start, ok = time.time(), True
    try:
        with open(test_file, 'rb') as f:
            mmapped_data = np.memmap(f, dtype=np.uint8, mode='r', shape=(total_size,))
            for i in range(0, total_size, chunk_size):
                chunk = mmapped_data[i:i + chunk_size]
                cp_data = cp.array(chunk, copy=False)
                comp_arr = codec.encode(nvcomp.Array(cp_data))
                ok &= bytes(chunk) == bytes(codec.decode(comp_arr).cpu())
                total_compressed += comp_arr.buffer_size
                del comp_arr
                cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        ok, total_compressed = False, 0
    return ok, total_size / total_compressed if total_compressed else 0, time.time() - start

def parse_size_input():
    while True:
        try:
            size_str = input("Enter file size (e.g., 256KB, 1MB, 2GB): ").strip().upper()
            if size_str.endswith('KB'):
                return int(size_str[:-2]) * 1024
            elif size_str.endswith('MB'):
                return int(size_str[:-2]) * 1024 * 1024
            elif size_str.endswith('GB'):
                return int(size_str[:-2]) * 1024 * 1024 * 1024
            return int(size_str)
        except ValueError:
            print("Invalid input. Use format: <number>[B|KB|MB|GB]")

def main():
    print("="*40, "COMPRESSION BENCHMARK", "="*40)
    get_system_info()
    file_size, test_file = parse_size_input(), "test_data.bin"
    if generate_test_data(test_file, file_size):
        cpu_ok, cpu_ratio, cpu_time = cpu_compression(test_file)
        print(f"CPU (pyzstd): {'OK' if cpu_ok else 'FAILED'}  Ratio: {cpu_ratio:.2f}x  Time: {cpu_time:.4f}s")
        gpu_ok, gpu_ratio, gpu_time = gpu_compression(test_file)
        print(f"GPU (nvcomp): {'OK' if gpu_ok else 'FAILED'}  Ratio: {gpu_ratio:.2f}x  Time: {gpu_time:.4f}s")
    if os.path.exists(test_file):
        os.remove(test_file)
    print("\nBenchmark complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
