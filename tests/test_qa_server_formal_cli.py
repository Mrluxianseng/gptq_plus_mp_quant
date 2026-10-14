import ast


def test_actual_preparser_retains_formal_run_and_hyperparameters():
    from tools.run_storage_formal_qa_mapped_v6 import source
    text=source();tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='p.parse_known_args')
    import argparse,sys
    old=sys.argv
    try:
        sys.argv=['entry','--runtime-root','/runtime','--component-root','/component','--run','--nsamples','2048','--seq-len','2048']
        scope={'argparse':argparse,'Path':__import__('pathlib').Path}
        exec(compile(ast.Module(body=fn.body[:end+1],type_ignores=[]),'actual_preparser_test','exec'),scope)
        assert scope['known'].runtime_root==__import__('pathlib').Path('/runtime')
        assert scope['remaining']==['--run','--nsamples','2048','--seq-len','2048']
    finally:sys.argv=old
