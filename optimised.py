import os
import time
import numpy as np
import asyncio
import cupy as cp
import pyzstd
from nvidia import nvcomp
from concurrent.futures import ThreadPoolExecutor

def get_system_info():
    import platform, sys, psutil
    info = {
        "Platform": platform.platform(),
        "Python": sys.version.split()[0],
        "Processor": platform.processor(),
        "Memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "GPU": cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        "GPU Memory": f"{cp.cuda.runtime.memGetInfo()[1] / (1024**3):.2f} GB"
    }
    print("System Info:", *[f"  {k}: {v}" for k, v in info.items()], sep="\n")

def generate_test_data(filename, size_bytes):
    chunk_size = 64 * 1024 * 1024  # 64MB chunks
    try:
        with open(filename, 'wb') as f:
            for _ in range(0, size_bytes, chunk_size):
                remaining = min(chunk_size, size_bytes - f.tell())
                f.write(bytes([i % 256 for i in range(remaining)]))
        return True
    except Exception as e:
        print(f"Error generating test data: {e}")
        return False

async def cpu_compression_async(test_file, executor, level=3):
    compressor = pyzstd.ZstdCompressor(level_or_option=level)  # Explicit level
    decompressor = pyzstd.ZstdDecompressor()
    total_size = os.path.getsize(test_file)
    
    async def compress_cpu(data):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, compressor.compress, data)

    try:
        with open(test_file, 'rb') as f:
            start = time.time()
            data = f.read()
            compressed = await compress_cpu(data)
            decompressed = await asyncio.to_thread(decompressor.decompress, compressed)
            ok = data == decompressed
        return ok, total_size / len(compressed) if compressed else 0, time.time() - start, len(compressed)
    except Exception as e:
        print(f"CPU compression error: {e}")
        return False, 0, time.time() - start, 0

async def gpu_compression_async(test_file, chunk_size_mb=None):
    codec = nvcomp.Codec(algorithm="Zstd")
    total_size = os.path.getsize(test_file)
    gpu_mem_free, gpu_mem_total = cp.cuda.runtime.memGetInfo()
    
    # Reserve 20% of GPU memory for compression overhead
    usable_gpu_mem = int(gpu_mem_free * 0.8)
    
    # Determine optimal chunk size based on available memory
    if chunk_size_mb is None:
        # If no specific chunk size, use adaptive sizing
        if total_size <= usable_gpu_mem // 3:  # Can fit whole file plus working space
            chunk_size = total_size
        else:
            # Use 1/3 of usable memory for each chunk to allow overlap
            chunk_size = usable_gpu_mem // 3
    else:
        chunk_size = chunk_size_mb * 1024 * 1024
        # Verify the requested chunk size isn't too large
        if chunk_size > usable_gpu_mem // 2:
            print(f"Warning: Requested chunk size may exceed GPU memory. Adjusting...")
            chunk_size = usable_gpu_mem // 3
    
    print(f"Using chunk size: {chunk_size/(1024*1024):.2f} MB")
    
    async def process_chunk(chunk_data, stream_id):
        try:
            # Create stream for this operation
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                # Correctly use the stream with cupy
                # First create the array
                cp_data = cp.asarray(chunk_data)
                # Then set the current stream and perform operations
                stream.use()
                
                # Create array and compress
                uncomp_array = nvcomp.Array(cp_data)
                comp_arrays = await asyncio.to_thread(codec.encode, [uncomp_array])
                
                if not comp_arrays or len(comp_arrays) != 1:
                    return False, 0
                
                comp_arr = comp_arrays[0]
                compressed_size = comp_arr.buffer_size
                
                # Decompress to verify
                decomp_arrays = await asyncio.to_thread(codec.decode, [comp_arr])
                
                if not decomp_arrays or len(decomp_arrays) != 1:
                    return False, compressed_size
                
                # Verify correctness
                decomp_arr = decomp_arrays[0]
                decomp_cpu = decomp_arr.cpu()
                ok = np.array_equal(decomp_cpu, chunk_data)
                
                # Clean up GPU memory immediately
                del cp_data, comp_arr, decomp_arr, uncomp_array
                stream.synchronize()
                
                # Periodically clean memory pool
                if stream_id % 2 == 0:
                    cp.get_default_memory_pool().free_all_blocks()
                
                return ok, compressed_size
        except Exception as e:
            print(f"Chunk processing error in stream {stream_id}: {e}")
            return False, 0

    try:
        total_compressed = 0
        start = time.time()
        
        # Use semaphore to limit concurrent GPU operations
        max_concurrent = 2  # Adjust based on your GPU
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(chunk_data, stream_id):
            async with semaphore:
                return await process_chunk(chunk_data, stream_id)
        
        with open(test_file, 'rb') as f:
            tasks = []
            stream_id = 0
            
            # Loop through file in chunks
            while chunk := f.read(chunk_size):
                chunk_data = np.frombuffer(chunk, dtype=np.uint8)
                tasks.append(bounded_process(chunk_data, stream_id))
                stream_id += 1
            
            # Process all chunks
            results = await asyncio.gather(*tasks)
            
            # Calculate total compressed size and check success
            for ok, size in results:
                if not ok:
                    return False, 0, time.time() - start, 0
                total_compressed += size

        # Calculate and return compression metrics
        elapsed = time.time() - start
        ratio = total_size / total_compressed if total_compressed else 0
        return True, ratio, elapsed, total_compressed
    
    except Exception as e:
        print(f"GPU compression error: {e}")
        cp.get_default_memory_pool().free_all_blocks()  # Emergency cleanup
        return False, 0, time.time() - start, 0
        
async def run_benchmark():
    print("="*40, "COMPRESSION BENCHMARK", "="*40)
    get_system_info()
    
    try:
        size_mb = int(input("Enter file size in MB: ").strip())
        chunk_size_mb = input("Enter GPU chunk size in MB (or press Enter for auto): ").strip()
        chunk_size_mb = int(chunk_size_mb) if chunk_size_mb else None
        
        size_bytes = size_mb * 1024 * 1024
        test_file = "test_data.bin"
        
        if not generate_test_data(test_file, size_bytes):
            print("Failed to generate test data")
            return
        
        with ThreadPoolExecutor() as executor:
            cpu_task = cpu_compression_async(test_file, executor)
            gpu_task = gpu_compression_async(test_file, chunk_size_mb)
            
            (cpu_ok, cpu_ratio, cpu_time, cpu_comp_size), \
            (gpu_ok, gpu_ratio, gpu_time, gpu_comp_size) = await asyncio.gather(cpu_task, gpu_task)
            
            cpu_throughput = size_mb / cpu_time if cpu_time > 0 else 0
            gpu_throughput = size_mb / gpu_time if gpu_time > 0 else 0
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            print(f"\nCPU (pyzstd): {'OK' if cpu_ok else 'FAILED'}")
            print(f"  Ratio: {cpu_ratio:.2f}x")
            print(f"  Compressed Size: {cpu_comp_size / (1024**2):.2f} MB")
            print(f"  Time: {cpu_time:.4f}s")
            print(f"  Throughput: {cpu_throughput:.2f} MB/s")
            
            print(f"GPU (nvcomp): {'OK' if gpu_ok else 'FAILED'}")
            print(f"  Ratio: {gpu_ratio:.2f}x")
            print(f"  Compressed Size: {gpu_comp_size / (1024**2):.2f} MB")
            print(f"  Time: {gpu_time:.4f}s")
            print(f"  Throughput: {gpu_throughput:.2f} MB/s")
            print(f"  Chunk Size: {chunk_size_mb if chunk_size_mb else 'Auto'} MB")
            print(f"Speedup (CPU/GPU): {speedup:.2f}x")
        
        if os.path.exists(test_file):
            os.remove(test_file)
            
    except ValueError:
        print("Invalid input: Enter numeric values for file and chunk sizes")
    except Exception as e:
        print(f"Benchmark error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Program error: {e}")