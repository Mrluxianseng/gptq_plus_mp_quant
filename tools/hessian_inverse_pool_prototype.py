"""Guarded CUDA solver-pool isolation; research only, no production defaults.

The normal allocator owns the already-consumed destination H group. All solver
temporaries use a separate native pool and must die before that pool is deleted.
Keep the exact helper arithmetic and factor strides. No CPU tensor transfer.
"""
import gc

from tools.hessian_inverse_lifetime_prototype import build_method


def install_isolated(implementation, *, enabled, records):
    torch = implementation.torch
    original = implementation.GPTQPlus._compute_hessian_inverse_with_fallback
    proposed, source = build_method(implementation)

    def inverse(self, h_sub, *args, **kwargs):
        target = str(getattr(self, "layer_name", "")).endswith("down_proj")
        if enabled and target:
            if self.alpha != 0 or kwargs.get("need_hinv_init", False):
                raise RuntimeError("solver-pool prototype requires alpha0 without retained inverse")
            subgroup = getattr(self, "_isolated_hessian_next_subgroup", 0)
            if subgroup >= self.num_groups:
                raise RuntimeError("solver-pool prototype requires one fasterquant pass per worker")
            storage = self.H[subgroup]
            if (not storage.is_cuda or storage.device != h_sub.device
                    or not storage.is_contiguous() or storage.shape != h_sub.shape
                    or storage.dtype != h_sub.dtype):
                raise RuntimeError("incompatible consumed group storage")
            pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(pool, device=h_sub.device):
                init, factor, damp, fallback = proposed(self, h_sub, *args, **kwargs)
                if init is not None:
                    raise RuntimeError("unexpected retained inverse")
                if factor.stride() not in ((1, storage.shape[0]), (storage.shape[0], 1)):
                    raise RuntimeError("unexpected solver factor stride")
                destination = storage.as_strided(factor.shape, factor.stride())
                destination.copy_(factor)
                del factor
            # Synchronization is deliberately explicit for the diagnostic
            # prototype; timing is not claimed until a low-overhead experiment.
            torch.cuda.synchronize(h_sub.device)

            def active_pool_bytes():
                snapshot = pool.snapshot(include_traces=False)
                return sum(block["size"] for segment in snapshot
                           for block in segment["blocks"]
                           if block["state"].startswith("active"))

            active_before_gc = active_pool_bytes()
            if active_before_gc:
                # A failed solver retry can leave an exception/traceback/frame
                # cycle. Collect only when real active pool blocks remain.
                gc.collect()
                torch.cuda.synchronize(h_sub.device)
            if active_pool_bytes():
                raise RuntimeError("solver tensor escaped private pool")
            # active_pool_bytes closes over pool, but holds no tensor/snapshot.
            del active_pool_bytes, pool
            self._isolated_hessian_next_subgroup = subgroup + 1
            result = init, destination, damp, fallback
        else:
            result = original(self, h_sub, *args, **kwargs)
            active_before_gc = 0
        if target:
            init, factor, damp, fallback = result
            records.append(dict(layer=getattr(self, "layer_idx", None),
                                module=self.layer_name, enabled=bool(enabled),
                                shape=list(h_sub.shape), input_stride=list(h_sub.stride()),
                                factor_stride=list(factor.stride()), dtype=str(h_sub.dtype),
                                damping=damp, identity_fallback=fallback,
                                inverse_retained=init is not None,
                                private_pool_released=bool(enabled),
                                active_pool_bytes_before_gc=active_before_gc))
        return result

    implementation.GPTQPlus._compute_hessian_inverse_with_fallback = inverse
    return source
