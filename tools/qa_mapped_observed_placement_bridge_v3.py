"""Versioned exclusive counter admission around the reviewed QA bridge.

No default launcher enablement. Prior native exact profiles must finish before
this interface starts, so outer interval counters are never silently reset.
"""
from tools.qa_mapped_observed_placement_bridge_v2 import MappedObservedPlacementBridge as Ancestor
from tools.qa_peak_counter_admission import CounterAdmission,verify_sources


class MappedObservedPlacementBridge(Ancestor):
    def __init__(self,*args,counter_source_bindings,**kwargs):
        if kwargs.get('validation_only') is not True:
            raise RuntimeError('Counter bridge is validation-only')
        verify_sources(counter_source_bindings)
        self.counter_source_bindings=dict(counter_source_bindings)
        super().__init__(*args,**kwargs)

    def qa_eval(self,model,tokenizer,*args,**kwargs):
        with CounterAdmission(self.root/'qa_counter_owner_rank0.json',
                bindings=self.counter_source_bindings,validation_only=True):
            return super().qa_eval(model,tokenizer,*args,**kwargs)
