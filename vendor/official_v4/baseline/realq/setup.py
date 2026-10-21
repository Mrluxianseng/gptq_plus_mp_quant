"""Install this directory as the realq package, including from a flat clone."""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements-core.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="real-q",
    version="0.1.0",
    description="End-to-end loss aligned LLM quantization with Block-GD",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=["realq"] + [
        "realq." + name
        for name in find_packages(where=str(ROOT), exclude=("build", "build.*", "tests", "tests.*"))
    ],
    package_dir={"realq": "."},
    package_data={"realq.utils": ["had_mat.pt"]},
    include_package_data=False,
    install_requires=requirements,
    entry_points={"console_scripts": ["realq=realq.ptq:main"]},
)

