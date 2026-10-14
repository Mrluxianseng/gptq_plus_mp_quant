"""Qwen3 scoring-only CPU weight offload; model/attention math unchanged."""
import torch


class LayerwiseWeights:
    def __init__(self,model,*,enabled,device,mark=lambda name:None,capture=lambda i,out:None):
        if model.config.model_type!='qwen3' or len(model.model.layers)!=model.config.num_hidden_layers:
            raise RuntimeError('Only complete Qwen3 decoder is supported')
        if any(hasattr(module,'_hf_hook') for module in model.modules()):
            raise RuntimeError('Do not combine with accelerate dispatch hooks')
        if model.training: raise RuntimeError('Scoring requires eval mode')
        self.model=model;self.enabled=bool(enabled);self.device=torch.device(device)
        self.mark=mark;self.capture=capture;self.handles=[];self.records=[];self.calls=[0]*len(model.model.layers)
        self.spec={name:(id(value),tuple(value.shape),str(value.dtype)) for name,value in model.named_parameters(remove_duplicate=False)}
        owners={}
        for index,layer in enumerate(model.model.layers):
            for value in layer.parameters():
                if id(value) in owners and owners[id(value)]!=index: raise RuntimeError('Cross-layer parameter alias is unsupported')
                owners[id(value)]=index
        for module in (model.model.embed_tokens,model.model.norm,model.model.rotary_emb,model.lm_head):
            if any(id(value) in owners for value in module.parameters()): raise RuntimeError('Anchor/layer alias is unsupported')
        if self.enabled:
            if any(value.device.type!='cpu' for layer in model.model.layers for value in layer.parameters()):
                raise RuntimeError('Layerwise installation requires CPU decoder weights')
            for module in (model.model.embed_tokens,model.model.norm,model.model.rotary_emb,model.lm_head): module.to(self.device)
        else: model.to(self.device)
        self.assert_identity()
        for index,layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self.pre(index)))
            self.handles.append(layer.register_forward_hook(self.post(index)))

    def assert_identity(self):
        current={name:(id(value),tuple(value.shape),str(value.dtype)) for name,value in self.model.named_parameters(remove_duplicate=False)}
        if current!=self.spec: raise RuntimeError('Parameter identity/shape/dtype changed')

    def pre(self,index):
        def hook(layer,inputs):
            self.calls[index]+=1
            self.mark(f'forward{self.calls[index]}_layer{index}_load_start')
            if self.enabled: layer.to(self.device)
            if any(value.device!=self.device for value in layer.parameters()): raise RuntimeError('Layer is not on execution device')
            self.mark(f'forward{self.calls[index]}_layer{index}_compute_start')
        return hook

    def post(self,index):
        def hook(layer,inputs,output):
            self.capture(index,output)
            self.mark(f'forward{self.calls[index]}_layer{index}_offload_start')
            if self.enabled: layer.to('cpu')
            expected=torch.device('cpu') if self.enabled else self.device
            if any(value.device!=expected for value in layer.parameters()): raise RuntimeError('Layer offload incomplete')
            self.records.append(dict(layer=index,forward=self.calls[index],enabled=self.enabled,
                parameter_bytes=sum(value.numel()*value.element_size() for value in layer.parameters()),
                final_device=str(expected)))
            self.mark(f'forward{self.calls[index]}_layer{index}_end')
        return hook

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles=[]
        if self.enabled:
            for layer in self.model.model.layers: layer.to('cpu')
        self.assert_identity()
