"""CPU-prepared early-runtime prefix only; does not enable formal mapped QA."""
import ast
import hashlib


SETUP='''
import os as _qa_early_os
import sys as _qa_early_sys
_qa_mapped_runtime_library=None
_qa_mapped_early_runtime={"formal_enabled":False,"prepared":False,
    "scope":"validation-only early runtime setup; not full QA acceptance"}
_qa_early_rank=int(_qa_early_os.environ.get("RANK","0"))
_qa_early_local_rank=int(_qa_early_os.environ.get("LOCAL_RANK","0"))
if _qa_early_rank<0 or _qa_early_local_rank<0:
    raise RuntimeError("Invalid distributed rank identity")
_qa_early_selected=(_qa_early_rank==0 and _qa_early_local_rank==0
    and "--exit_after_precompute" not in _qa_early_sys.argv
    and "--lm_eval" in _qa_early_sys.argv and "--skip_eval" not in _qa_early_sys.argv)
_qa_mapped_early_runtime.update(rank=_qa_early_rank,local_rank=_qa_early_local_rank,selected=_qa_early_selected)
if _qa_early_selected:
    if _qa_early_os.environ.get("REALQ_QA_STORAGE_VALIDATION")!="1":
        raise RuntimeError("Prepared mapped entry requires explicit validation-only environment")
    from pathlib import Path as _qa_early_Path
    import hashlib as _qa_early_hashlib
    _qa_early_binary=_qa_early_Path(_qa_early_os.environ["QA_MAPPED_HEAD_LIBRARY"])
    _qa_early_digest=_qa_early_hashlib.sha256(_qa_early_binary.read_bytes()).hexdigest()
    if _qa_early_digest!=_qa_early_os.environ["QA_MAPPED_HEAD_LIBRARY_SHA256"]:
        raise RuntimeError("Admitted mapped runtime binary changed")
    from tools.qa_mapped_head_storage_prototype import prepare_runtime as _qa_early_prepare
    _qa_mapped_runtime_library,_qa_early_capabilities=_qa_early_prepare(_qa_early_binary)
    _qa_mapped_early_runtime.update(prepared=True,binary=str(_qa_early_binary),
        binary_sha256=_qa_early_digest,capabilities=_qa_early_capabilities)
'''


def build_entry(original_source,*,validation_only):
    if validation_only is not True: raise RuntimeError('This entry is preparation/validation only')
    tree=ast.parse(original_source)
    runs=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and ast.unparse(node.func)=='runpy.run_path']
    if len(runs)!=1 or not runs[0].args or not isinstance(runs[0].args[0],ast.Constant) or runs[0].args[0].value!='ptq.py':
        raise RuntimeError('Requires the unique original ptq entry')
    if any(isinstance(node,ast.Name) and node.id.startswith('_qa_early_') for node in ast.walk(tree)):
        raise RuntimeError('Early runtime namespace already installed or colliding')
    index=0
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str): index=1
    while index<len(tree.body) and isinstance(tree.body[index],ast.ImportFrom) and tree.body[index].module=='__future__': index+=1
    prefix=ast.parse(SETUP).body
    tree.body[index:index]=prefix;ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+'\n';compile(result,'qa_mapped_validation_early_entry','exec')
    return result,dict(formal_enabled=False,validation_only=True,setup_statement_count=len(prefix),
        original_source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(result.encode()).hexdigest())
