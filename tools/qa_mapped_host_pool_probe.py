"""Runtime/memory-accounting and tiny whole-GEMM probe; NOT a QA method yet."""
import argparse
import ctypes
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    import torch
    from tools.qa_tensor_fingerprint_stream import tensor_sha256
    libdir=Path(torch.__file__).parent.parent/'nvidia/cuda_runtime/lib'
    runtime=libdir/'libcudart.so.12'
    source=Path(__file__).with_suffix('.cpp').with_name('qa_mapped_host_allocator_probe.cpp')
    binary=root/'mapped_host_probe.so'
    command=['g++','-std=c++17','-O2','-shared','-fPIC','-pthread',str(source),str(runtime),
        '-Wl,-rpath,'+str(libdir),'-o',str(binary)]
    with (root/'compile.log').open('x') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    (root/'compile_command.json').write_text(json.dumps(command,indent=2))
    headers=runtime.parent.parent/'include'
    files=[Path(__file__),source,binary,runtime,headers/'driver_types.h',headers/'cuda_runtime_api.h',
        Path(torch.__file__).parent/'include/torch/csrc/cuda/CUDAPluggableAllocator.h',
        Path(inspect.getfile(torch.cuda.memory.MemPool)),Path('tools/qa_tensor_fingerprint_stream.py')]
    bindings={str(p.resolve()):sha(p) for p in files}
    (root/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    library=ctypes.CDLL(str(binary.resolve()))
    for name in ('qa_live_bytes','qa_peak_bytes','qa_allocation_count','qa_release_count'):
        getattr(library,name).restype=ctypes.c_size_t
    record=dict(scope='small runtime/native-accounting/whole-GEMM probe only; no model/QA capacity claim',valid=False,
        torch_version=str(torch.__version__),torch_cuda=torch.version.cuda)
    pool=None;tensor=None;result=None
    try:
        if torch.cuda.is_initialized(): raise RuntimeError('Map-host flag probe requires a fresh CUDA process')
        record['map_host_flag_error']=library.qa_enable_map_host()
        if record['map_host_flag_error']!=0: raise RuntimeError('Cannot enable map-host context flag')
        record['attributes']={name:library.qa_device_attribute(number,0) for name,number in
            (('can_map_host_memory',19),('unified_addressing',41))}
        if any(value!=1 for value in record['attributes'].values()): raise RuntimeError('Mapped-host runtime capability missing')
        torch.cuda.init()
        allocator=torch.cuda.memory.CUDAPluggableAllocator(str(binary.resolve()),'qa_mapped_alloc','qa_mapped_free')
        pool=torch.cuda.MemPool(allocator.allocator())
        before=dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0))
        with torch.cuda.use_mem_pool(pool):
            tensor=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda:0')
            tensor.fill_(7)
        torch.cuda.synchronize()
        snapshots=pool.snapshot(include_traces=False)
        active=sum(segment['active_size'] for segment in snapshots)
        reserved=sum(segment['total_size'] for segment in snapshots)
        record['basic']=dict(payload=tensor.numel()*tensor.element_size(),private_active=active,private_reserved=reserved,
            allocated_delta=torch.cuda.memory_allocated(0)-before['allocated'],
            reserved_delta=torch.cuda.memory_reserved(0)-before['reserved'],mapped_host_live=library.qa_live_bytes(),
            cuda_copy_sha=tensor_sha256(tensor),expected_sha=hashlib.sha256(bytes([7])*(64*1024*1024)).hexdigest())
        record['native_accounting_complete']=(record['basic']['allocated_delta']>=record['basic']['payload']
            and record['basic']['reserved_delta']>=reserved and active>=record['basic']['payload']
            and library.qa_live_bytes()==reserved)
        if not record['native_accounting_complete']:
            raise RuntimeError('Mapped private pool is missing native allocated/reserved coverage; reject progression')
        if record['basic']['cuda_copy_sha']!=record['basic']['expected_sha']:
            raise RuntimeError('Mapped CUDA tensor byte mismatch')
        del tensor;tensor=None;gc.collect();torch.cuda.synchronize()
        record['basic_released_active']=sum(s['active_size'] for s in pool.snapshot(include_traces=False))
        if record['basic_released_active']!=0: raise RuntimeError('Private tensor still active')
        torch.manual_seed(1)
        inputs=torch.randn(16,32,256,device='cuda:0',dtype=torch.bfloat16)
        weight=torch.randn(8192,256,device='cuda:0',dtype=torch.bfloat16)
        with torch.no_grad():
            reference=torch.nn.functional.linear(inputs,weight)
            reference_sha=tensor_sha256(reference)
            with torch.cuda.use_mem_pool(pool): result=torch.nn.functional.linear(inputs,weight)
            torch.cuda.synchronize()
            record['gemm']=dict(input_shape=list(inputs.shape),weight_shape=list(weight.shape),output_shape=list(result.shape),
                dtype=str(result.dtype),stride=list(result.stride()),full_reference_sha=reference_sha,
                full_candidate_sha=tensor_sha256(result),exact=torch.equal(reference,result))
        if record['gemm']['full_reference_sha']!=record['gemm']['full_candidate_sha']:
            raise RuntimeError('Whole GEMM mapped-output byte mismatch')
        del result;result=None;del inputs,weight,reference;gc.collect();torch.cuda.synchronize()
        record['final_active']=sum(s['active_size'] for s in pool.snapshot(include_traces=False))
        if record['final_active']!=0: raise RuntimeError('Private pool retains active storage')
        del pool;pool=None;gc.collect();torch.cuda.synchronize()
        record['final_host_live_bytes']=library.qa_live_bytes()
        record['allocations']=library.qa_allocation_count();record['releases']=library.qa_release_count()
        record['valid']=(record['final_host_live_bytes']==0 and record['allocations']==record['releases']
            and library.qa_last_error()==0 and all(sha(p)==h for p,h in bindings.items()))
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),allocator_error=library.qa_last_error())
    finally:
        tensor=None;result=None;gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.synchronize()
        if pool is not None:
            record['failure_cleanup_active']=sum(s['active_size'] for s in pool.snapshot(include_traces=False))
            if record['failure_cleanup_active']==0:
                pool=None;gc.collect()
                record['failure_cleanup_host_live_bytes']=library.qa_live_bytes()
        with (root/'status.json').open('x') as stream: json.dump(record,stream,indent=2)
        print(json.dumps(record,indent=2),flush=True)
    if not record['valid']: raise SystemExit(1)


if __name__=='__main__': main()
