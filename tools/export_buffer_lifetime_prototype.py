"""Research-only delayed allocation of optional GPTQ integer export arrays.

Only full/subgroup code and scale *copies* are omitted when no export or
diagnostic consumer exists. Column quantization and its block code/scale
workspaces remain unchanged, including the fused-kernel API.
"""
import inspect
import textwrap


def transform_source(source):
    source = textwrap.dedent(source)
    changes = {
        "rec = diagnostic_recorder": "rec = diagnostic_recorder\n    _retain_export_copies = bool(export_to_et or rec is not None)",
        "W_int_final = torch.zeros_like(W)": "W_int_final = torch.zeros_like(W) if _retain_export_copies else None",
        "Scale_final = torch.zeros_like(W)": "Scale_final = torch.zeros_like(W) if _retain_export_copies else None",
        "W_int_sub = torch.zeros_like(W_sub)": "W_int_sub = torch.zeros_like(W_sub) if _retain_export_copies else None",
        "Scale_sub = torch.zeros_like(W_sub)": "Scale_sub = torch.zeros_like(W_sub) if _retain_export_copies else None",
        'Scale1 = torch.zeros_like(W1).to(state["Scale_sub"].dtype)': 'Scale1 = torch.zeros_like(W1).to(state["Scale_sub"].dtype if _retain_export_copies else W1.dtype)',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise RuntimeError(f"export prototype source boundary changed: {before}")
        source = source.replace(before, after)
    guarded = (
        'state["W_int_sub"][:, i1:i2] = W_int1',
        'state["Scale_sub"][:, i1:i2] = Scale1',
        'W_int_sub = W_int_sub[:, state["invperm"]]',
        'Scale_sub = Scale_sub[:, state["invperm"]]',
        'W_int_final[state["row_start"]:state["row_end"], :] = W_int_sub',
        'Scale_final[state["row_start"]:state["row_end"], :] = Scale_sub',
    )
    lines = source.splitlines()
    for statement in guarded:
        matching = [index for index, line in enumerate(lines) if line.strip() == statement]
        if len(matching) != 1:
            raise RuntimeError(f"export copy consumer boundary changed: {statement}")
        index = matching[0]
        indent = lines[index][:len(lines[index])-len(lines[index].lstrip())]
        lines[index:index+1] = [indent + "if _retain_export_copies:", indent + "    " + statement]
    result = "\n".join(lines)+"\n"
    compile(result, "lazy_export_fasterquant", "exec")
    return result


def build_method(impl):
    source = transform_source(inspect.getsource(impl.GPTQPlus.fasterquant))
    namespace = dict(vars(impl))
    exec(compile(source, "lazy_export_fasterquant", "exec"), namespace)
    return namespace["fasterquant"], source
