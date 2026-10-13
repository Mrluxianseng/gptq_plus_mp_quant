"""Allocator maxima for every visible device used by one process, including QA."""
from __future__ import annotations
import os

SCHEMA = 'all-visible-device-maxima-v1'


def collect_peaks(cuda, getter):
    devices = []
    tokens = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    for index in range(cuda.device_count()):
        allocated, reserved = getter(f'cuda:{index}')
        # Querying allocator counters does not require synchronizing unused GPUs.
        if allocated or reserved:
            cuda.synchronize(index)
            allocated, reserved = getter(f'cuda:{index}')
        devices.append(dict(device_index=index, physical_selector=tokens[index] if index<len(tokens) and tokens[index] else str(index),
                            peak_allocated_bytes=int(allocated), peak_reserved_bytes=int(reserved)))
    return dict(device_peak_schema=SCHEMA, devices=devices,
                visible_device_count=len(devices),
                peak_allocated_bytes=max((x['peak_allocated_bytes'] for x in devices), default=0),
                peak_reserved_bytes=max((x['peak_reserved_bytes'] for x in devices), default=0))


def valid_record(row, visible_devices):
    devices = row.get('devices', [])
    if (row.get('device_peak_schema')!=SCHEMA or row.get('visible_device_count')!=visible_devices
            or len(devices)!=visible_devices or {x.get('device_index') for x in devices}!=set(range(visible_devices))):
        return False
    if any(any(type(x.get(k)) is not int or x[k]<0 for k in ('peak_allocated_bytes','peak_reserved_bytes')) for x in devices):
        return False
    return all(row.get(k)==max(x[k] for x in devices) for k in ('peak_allocated_bytes','peak_reserved_bytes'))
