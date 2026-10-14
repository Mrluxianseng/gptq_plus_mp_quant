"""Validation-only composition of placement, complete observation and mapped head.

Current head admission requires the original Linear forward (no accelerate
head hook). This is a local single-device compatibility bridge, not a formal
multi-device launcher. The runtime must be prepared before CUDA initialization.
"""
import ast
import hashlib
import inspect
import json
import marshal
from tools.qa_observed_placement_bridge import ObservedPlacementBridge
from tools.qa_score_lifetime_prototype import build_method as rows_method
from tools.qa_mapped_head_natural_batch import NaturalBatchMappedHeadStorage,build_batch_method


def selected_digest(method):
    return hashlib.sha256(marshal.dumps(inspect.unwrap(method).__code__)).hexdigest()


def control_batch_source(original_source):
    import textwrap
    tree=ast.parse(textwrap.dedent(original_source))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!='_loglikelihood_tokens':
        raise RuntimeError('Original native score source changed')
    batches=[n for n in ast.walk(functions[0]) if isinstance(n,ast.For)
        and ast.unparse(n.target)=='chunk' and ast.unparse(n.iter)=='chunks']
    if len(batches)!=1: raise RuntimeError('Original native batch-loop changed')
    batches[0].body.append(ast.parse('_qa_release_batch()').body[0])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


def control_batch_method(original,release):
    scope=dict(original.__globals__,_qa_release_batch=release)
    exec(compile(control_batch_source(inspect.getsource(original)),'qa_native_batch_observer','exec'),scope)
    return scope['_loglikelihood_tokens']


class MappedObservedPlacementBridge(ObservedPlacementBridge):
    def __init__(self,*args,mapped_enabled,library,allocator=None,**kwargs):
        from lm_eval.models.huggingface import HFLM
        if mapped_enabled and (not kwargs.get('enabled') or kwargs.get('device','cuda:0')=='cpu'):
            raise RuntimeError('Mapped candidate requires admitted CUDA layerwise placement')
        if mapped_enabled and allocator is None: raise RuntimeError('Admitted mapped allocator required')
        self.mapped_enabled=bool(mapped_enabled);self.library=library;self.allocator=allocator
        self.native_score=HFLM._loglikelihood_tokens
        self.head_ticket=None;self.failed_ticket=None
        super().__init__(*args,**kwargs)
        self.mapped_path=self.root/'qa_mapped_head_rank0.json'
        if self.mapped_path.exists():
            self.restore();raise RuntimeError('Mapped evidence exists; choose a fresh directory')

    def observed_eval(self,model,tokenizer,*args,**kwargs):
        from lm_eval.models.huggingface import HFLM
        installed=HFLM._loglikelihood_tokens
        expected=rows_method(self.native_score) if self.bridge.enabled else self.native_score
        if selected_digest(installed)!=selected_digest(expected):
            raise RuntimeError('Do not stack an unreviewed selected score method')
        primary=None;record=dict(completed=False,formal_enabled=False,mapped_enabled=self.mapped_enabled,
            scope='validation-only local original-Linear composition, not formal QA/multi-device admission',
            installed_score_code_sha256=selected_digest(installed))
        ticket=None
        try:
            ticket=NaturalBatchMappedHeadStorage(model.lm_head,configured_batch_size=16,
                enabled=self.mapped_enabled,expected_shape=(model.config.vocab_size,model.config.hidden_size),
                library=self.library,cuda=self.cuda,require_cuda=self.bridge.device.type=='cuda',
                pool_factory=lambda:self.cuda.MemPool(self.allocator.allocator()),mark=self.mark)
            self.head_ticket=ticket
            HFLM._loglikelihood_tokens=(build_batch_method if self.bridge.enabled else control_batch_method)(
                self.native_score,ticket.release_batch)
            record['selected_score_code_sha256']=selected_digest(HFLM._loglikelihood_tokens)
            result=super().observed_eval(model,tokenizer,*args,**kwargs)
            record['completed']=True
            return result
        except BaseException as exc:
            primary=exc;record.update(completed=False,error_type=type(exc).__name__,error=str(exc));raise
        finally:
            HFLM._loglikelihood_tokens=installed
            if ticket is not None:
                try:
                    self.mark('qa_mapped_storage_cleanup')
                    record['head']=ticket.close()
                    record['callback_host_peak_bytes']=int(self.library.qa_peak_bytes()) if self.mapped_enabled else 0
                except BaseException as cleanup:
                    record.update(completed=False,cleanup_error_type=type(cleanup).__name__,cleanup_error=str(cleanup))
                    for handle in ticket.handles: handle.remove()
                    # Preserve the pool owner if a failing traceback still
                    # owns an output. Never delete an active mapped pool.
                    self.failed_ticket=ticket
                    if primary is None: primary=cleanup
            self.head_ticket=None
            self.root.mkdir(parents=True,exist_ok=True)
            with self.mapped_path.open('x') as stream: json.dump(record,stream,indent=2,allow_nan=False)
            if record.get('cleanup_error') and record.get('error_type') is None: raise primary
