"""Release mapped output segments after the scored batch's last consumer."""
import ast
import gc
import inspect
from tools.qa_mapped_head_storage_prototype import MappedHeadStorage,prepare_runtime
from tools.qa_score_lifetime_prototype import build_source as rows_source


class BatchMappedHeadStorage(MappedHeadStorage):
    def __init__(self,head,*,pool_factory,**kwargs):
        self.pool_factory=pool_factory
        self.batch_releases=[]
        enabled=kwargs['enabled']
        super().__init__(head,pool=pool_factory() if enabled else None,**kwargs)

    def pre(self,module,inputs):
        if self.enabled and self.pool is None: self.pool=self.pool_factory()
        return super().pre(module,inputs)

    def release_batch(self):
        if self.context is not None or self.closed or self.calls!=len(self.batch_releases)+1:
            raise RuntimeError('Batch release must follow exactly one completed head forward')
        if not self.records or not self.records[-1]['completed']:
            raise RuntimeError('Failed head cannot be accepted as a completed batch')
        self.mark(f'head{self.calls}_storage_release_start')
        record=dict(call=self.calls,enabled=self.enabled)
        if self.enabled:
            self.cuda.synchronize()
            active=sum(segment['active_size'] for segment in self.pool.snapshot(include_traces=False))
            record['active_before_release']=active
            if active!=0: raise RuntimeError('Mapped batch output still has a consumer; do not delete pool')
            self.pool=None;gc.collect();self.cuda.synchronize()
            record.update(host_live_after_release=int(self.library.qa_live_bytes()),
                allocations=int(self.library.qa_allocation_count()),releases=int(self.library.qa_release_count()))
            if record['host_live_after_release']!=0 or record['allocations']!=record['releases'] or self.library.qa_last_error()!=0:
                raise RuntimeError('Batch mapped pool release/accounting incomplete')
        self.batch_releases.append(record)
        self.mark(f'head{self.calls}_storage_release_end')

    def close(self):
        if len(self.batch_releases)!=self.calls or self.pool is not None:
            raise RuntimeError('Every scored batch must release its pool at the last consumer')
        enabled=self.enabled
        # Parent restores hooks/validates Parameter identity. Each private pool
        # is already gone; use the parent's no-pool cleanup branch only.
        self.enabled=False
        try: result=super().close()
        finally: self.enabled=enabled
        result.update(enabled=enabled,batch_releases=self.batch_releases)
        if enabled:
            result.update(released_active=0,final_host_live=int(self.library.qa_live_bytes()),
                allocations=int(self.library.qa_allocation_count()),releases=int(self.library.qa_release_count()))
            if result['final_host_live']!=0 or result['allocations']!=result['releases']:
                raise RuntimeError('Final mapped callback accounting incomplete')
        return result


def build_batch_source(original_source):
    tree=ast.parse(rows_source(original_source))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    fn=functions[0]
    matches=[node for node in ast.walk(fn) if isinstance(node,ast.Delete)
        and ast.unparse(node)=='del multi_logits']
    if len(matches)!=1: raise RuntimeError('Original rows last-consumer deletion contract changed')
    target=matches[0]
    class Insert(ast.NodeTransformer):
        def visit_Delete(self,node):
            return [node,ast.parse('_qa_release_batch()').body[0]] if node is target else node
    Insert().visit(fn);ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+'\n';compile(result,'qa_mapped_batch_lifetime','exec')
    return result


def build_batch_method(original,release):
    scope=dict(original.__globals__,_qa_release_batch=release)
    exec(compile(build_batch_source(inspect.getsource(original)),'qa_mapped_batch_lifetime','exec'),scope)
    return scope['_loglikelihood_tokens']
