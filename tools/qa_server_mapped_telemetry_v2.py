"""Versioned safe snapshot reader and durable head-accounting journal.

Frozen allocator, original Linear/GEMM, scores, placement and peak guards are
unchanged. Old failed artifacts remain failures. This is opt-in validation.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import textwrap
from tools import qa_mapped_observed_placement_bridge as base_bridge
from tools import qa_mapped_observed_placement_bridge_v2 as control_bridge
from tools import qa_mapped_observed_placement_bridge_v3 as owner_bridge
from tools import qa_server_mapped_validation as server
from tools import qa_server_mapped_telemetry as telemetry
from tools.qa_mapped_head_natural_batch import NaturalBatchMappedHeadStorage
from tools.qa_server_runtime_compat_v2 import SnapshotCUDA


class JournaledHead(NaturalBatchMappedHeadStorage):
    def __init__(self, head, *, journal, **kwargs):
        self.journal = Path(journal)
        if self.journal.exists():
            raise FileExistsError('Preserve original head accounting journal')
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open('x'):
            pass
        super().__init__(head, **kwargs)

    def post(self, module, inputs, output):
        previous = len(self.records)
        try:
            return super().post(module, inputs, output)
        finally:
            if len(self.records) > previous:
                row = dict(self.records[-1])
                row['current_device'] = self.cuda.current_device()
                row['output_device'] = None if output is None else str(output.device)
                row['output_data_ptr'] = None if output is None else output.data_ptr()
                if self.enabled:
                    row['pool_id'] = list(self.pool.id)
                    row['private_segments'] = self.pool.snapshot(include_traces=False)
                    row['native_segments'] = self.cuda.memory_snapshot(include_traces=False)
                    row['callback_error'] = int(self.library.qa_last_error())
                    row['callback_allocations'] = int(self.library.qa_allocation_count())
                    row['callback_releases'] = int(self.library.qa_release_count())
                with self.journal.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    if row.get('accounting_complete') is False:
                        os.fsync(stream.fileno())


def head_ticket(head, **kwargs):
    if kwargs['enabled']:
        mark = kwargs['mark']
        owner = getattr(mark, '__self__', None)
        if owner is None or not hasattr(owner, 'root'):
            raise RuntimeError('Head journal requires the actual bridge destination')
        return JournaledHead(head, journal=owner.root/'qa_head_accounting_journal.jsonl', **kwargs)
    return control_bridge.head_ticket(head, **kwargs)


def replace_base(class_source, old, new):
    tree = ast.parse(textwrap.dedent(class_source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise RuntimeError('Expected one reviewed bridge class')
    node = tree.body[0]
    if len(node.bases) != 1 or ast.unparse(node.bases[0]) != old:
        raise RuntimeError('Reviewed bridge inheritance changed')
    node.bases = [ast.Name(id=new, ctx=ast.Load())]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# Reconstruct each reviewed layer into a private namespace. Do not monkeypatch
# frozen modules or stack a second math/lifetime observer on their methods.
namespace = dict(base_bridge.__dict__, _qa_admitted_head_ticket=head_ticket)
exec(compile(control_bridge.bridge_source(), 'qa_safe_snapshot_composition', 'exec'), namespace)
reviewed_base = namespace['MappedObservedPlacementBridge']
owner_scope = dict(owner_bridge.__dict__, Ancestor=reviewed_base)
exec(compile(inspect.getsource(owner_bridge.MappedObservedPlacementBridge),
             'qa_safe_snapshot_counter_owner', 'exec'), owner_scope)
server_scope = dict(server.__dict__, MappedObservedPlacementBridge=owner_scope['MappedObservedPlacementBridge'])
exec(compile(inspect.getsource(server.ServerBridge), 'qa_safe_snapshot_actual_cache', 'exec'), server_scope)
telemetry_scope = dict(telemetry.__dict__, SafeServerBridge=server_scope['ServerBridge'])
exec(compile(replace_base(inspect.getsource(telemetry.TelemetryServerBridge),
                         'ancestor.ServerBridge', 'SafeServerBridge'),
             'qa_safe_snapshot_telemetry', 'exec'), telemetry_scope)
TelemetryServerBridge = telemetry_scope['TelemetryServerBridge']
install_scope = dict(server.__dict__, ServerBridge=TelemetryServerBridge, SnapshotCUDA=SnapshotCUDA)
exec(compile(inspect.getsource(server.install), 'qa_safe_snapshot_install', 'exec'), install_scope)
install = install_scope['install']
