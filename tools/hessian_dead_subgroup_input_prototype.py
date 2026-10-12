"""Release the last private subgroup input before quantization/refresh.

The sequential solver loop retains its final H_sub local until fasterquant
returns, despite its last consumer having finished. Leave immutable factors
in subgroup_states intact. Preparation only: real CUDA trajectory is required.
"""
import ast


RELEASE_BLOCK = '''if (str(getattr(self, "layer_name", "")).endswith("down_proj")
        and self.alpha == 0 and not hessian_cpu_offload):
    # All solver/diagnostic input consumers have completed. Guard against
    # accidentally freeing a factor alias after a future helper refactor.
    _dead_h_ptr = H_sub.untyped_storage().data_ptr()
    if any(state["Hinv"] is not None
            and state["Hinv"].untyped_storage().data_ptr() == _dead_h_ptr
            for state in subgroup_states):
        raise RuntimeError("last subgroup input aliases a retained factor")
    _dead_h_bytes = H_sub.numel() * H_sub.element_size()
    del H_sub
    _dead_h_records = getattr(self, "_dead_subgroup_input_records", None)
    if _dead_h_records is not None:
        _dead_h_records.append(dict(layer=getattr(self, "layer_idx", None),
            module=self.layer_name, released_private_input_bytes=_dead_h_bytes,
            retained_factors=len(subgroup_states)))
'''


def build_source(source):
    """Accept explicit, bound generated fasterquant source, not an observer."""
    anchor = "# Number of blocks + refreshes, for the loss-slide-window schedule."
    lines = [line for line in source.splitlines() if line.strip() == anchor]
    if len(lines) != 1:
        raise RuntimeError("subgroup setup/quantization boundary changed")
    line = lines[0]
    before, after = source.split(line, 1)
    # Never silently remove a local with additional downstream consumers.
    tree = ast.parse(source)
    boundary_line = len(before.splitlines()) + 1
    if any(isinstance(node, ast.Name) and node.id == "H_sub"
           and node.lineno >= boundary_line for node in ast.walk(tree)):
        raise RuntimeError("H_sub still has downstream consumers")
    indent = line[:len(line) - len(line.lstrip())]
    block = "\n".join(indent + text if text else "" for text in RELEASE_BLOCK.splitlines())
    result = before + block + "\n" + line + after
    compile(result, "hessian_dead_subgroup_input_prototype", "exec")
    return result
