"""Fresh CUDA check for solver out= reuse; no full-model quality claim."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import check_hessian_stable_weight_cuda as check
from tools.hessian_solver_workspace_prototype import build_method,install_down_proj


def main():
    check.fixture.build_method=build_method
    check.fixture.install_down_proj=install_down_proj
    check.main()
    out=Path(sys.argv[sys.argv.index("--output-directory")+1])
    report=json.loads((out/"status.json").read_text())
    component_valid=report["valid"]
    report["valid"]=False
    (out/"status.json").write_text(json.dumps(report,indent=2))
    helper=ROOT/"tools/hessian_solver_workspace_prototype.py"
    (out/"workspace_prototype.py").write_bytes(helper.read_bytes())
    report["workspace_source_sha256"]=hashlib.sha256(helper.read_bytes()).hexdigest()
    report["valid"]=component_valid and all(
        len(m["api"]["arms"][1]["calls"])==4 and
        all(c.get("workspace_reuse_enabled") for c in m["api"]["arms"][1]["calls"])
        for m in report["models"])
    (out/"status.json").write_text(json.dumps(report,indent=2))
    print('[solver-workspace] bound CUDA preflight valid='+str(report["valid"]),flush=True)
    if not report["valid"]:
        raise RuntimeError("workspace reuse coverage failed")


if __name__=='__main__':
    main()
