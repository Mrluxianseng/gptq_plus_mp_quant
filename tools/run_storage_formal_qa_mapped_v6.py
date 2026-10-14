"""Disable argparse abbreviation so --run cannot consume --runtime-root."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_storage_formal_qa_mapped_v5 import source as prior_source


def source():
    text=prior_source()
    old='p=argparse.ArgumentParser(add_help=False)'
    if text.count(old)!=1:raise RuntimeError('Frozen formal argument pre-parser changed')
    text=text.replace(old,'p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)')
    old="extra=[ROOT/'tools/run_storage_formal_qa_mapped_v4.py',"
    if text.count(old)!=1:raise RuntimeError('Frozen v5 source manifest changed')
    return text.replace(old,"extra=[ROOT/'tools/run_storage_formal_qa_mapped_v5.py',ROOT/'tools/run_storage_formal_qa_mapped_v4.py',")


scope=dict(__name__='qa_server_formal_v6',__file__=__file__)
exec(compile(source(),'qa_server_formal_v6','exec'),scope)
if __name__=='__main__':scope['main']()
