"""Preserve original worker text anchors when inserting early runtime setup."""
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_storage_formal_qa_mapped_v7 import source as prior_source
from tools.qa_mapped_early_runtime_entry import SETUP as EARLY_SETUP


def source():
    text=prior_source()
    manifest="manifest=known.runtime_root/'formal_server_bindings.json'"
    if text.count(manifest)!=1:raise RuntimeError('Frozen admission manifest location changed')
    text=text.replace(manifest,"manifest=known.runtime_root/(formal_args.output_directory.name+'_admitted_sources.json')")
    anchor="extra=[ROOT/'tools/run_storage_formal_qa_mapped_v6.py',"
    if text.count(anchor)!=1:raise RuntimeError('Frozen v7 source manifest changed')
    return text.replace(anchor,"extra=[ROOT/'tools/run_storage_formal_qa_mapped_v7.py',ROOT/'tools/run_storage_formal_qa_mapped_v6.py',")


scope=dict(__name__='qa_server_formal_v8',__file__=__file__)
exec(compile(source(),'qa_server_formal_v8','exec'),scope)
prior_entry=scope['entry']


def entry(original):
    # Preserve the old AST contracts but avoid reformatting frozen worker text:
    # the production profiling installer deliberately matches exact anchors.
    prior_entry(original)
    anchor="runpy.run_path('ptq.py', run_name='__main__')"
    tree=ast.parse(original);position=0;index=0
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
        position=tree.body[0].end_lineno;index=1
    while index<len(tree.body) and isinstance(tree.body[index],ast.ImportFrom) and tree.body[index].module=='__future__':
        position=tree.body[index].end_lineno;index+=1
    lines=original.splitlines(keepends=True);lines[position:position]=[EARLY_SETUP+'\n']
    result=''.join(lines).replace(anchor,scope['SETUP']+'\n'+anchor)
    compile(result,'preserved_formal_QA_worker','exec')
    return result


scope['entry']=entry
if __name__=='__main__':scope['main']()
