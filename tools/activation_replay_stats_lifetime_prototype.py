"""Prepare last-consumer cleanup of quantization statistics before replay.

No CUDA experiment or patch to the production module is performed here.
Move existing deletes only; replay inputs, model parameters and arithmetic
are untouched. A generated full entry still needs independent acceptance.
"""
import ast


NAMES = frozenset(("gptq", "saliency_dict", "gradients_dict"))


def build_source(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "gptq_fwrd":
        raise RuntimeError("expected explicit bound gptq_fwrd source")
    lines = source.splitlines(keepends=True)
    marker = 'memory_profiler.mark(f"layer_{i}_activation_replay_and_cleanup", layer=i)'
    matches = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(matches) != 1:
        raise RuntimeError("replay boundary changed")
    mark_index = matches[0]
    if mark_index == 0 or lines[mark_index - 1].strip() != "if memory_profiler is not None:":
        raise RuntimeError("replay marker guard changed")
    insertion = mark_index - 1
    indent = lines[insertion][:len(lines[insertion]) - len(lines[insertion].lstrip())]
    deletions = []
    for text in ("del gptq", "del saliency_dict, gradients_dict"):
        found = [j for j, line in enumerate(lines) if line.strip() == text]
        if len(found) != 1 or found[0] <= mark_index:
            raise RuntimeError("late statistics cleanup boundary changed")
        line = lines[found[0]]
        if line[:len(line) - len(line.lstrip())] != indent + "    ":
            raise RuntimeError("expected original deletes inside layer.cleanup")
        deletions.append(found[0])
    # Reject a new consumer or rebinding introduced after the old last consumer.
    # Name deletion itself is expected at the old layer.cleanup site.
    if any(isinstance(node, ast.Name) and node.id in NAMES
           and node.lineno >= insertion + 1 and not isinstance(node.ctx, ast.Del)
           for node in ast.walk(tree)):
        raise RuntimeError("quantization statistics have a downstream consumer")
    for j in sorted(deletions, reverse=True):
        del lines[j]
    lines[insertion:insertion] = [
        indent + "# Last consumers finished, including after_layer_quantized hook.\n",
        indent + "# Drop only private quantization statistics before activation replay.\n",
        indent + "del gptq\n",
        indent + "del saliency_dict, gradients_dict\n",
    ]
    result = "".join(lines)
    compile(result, "activation_replay_stats_lifetime_prototype", "exec")
    return result
