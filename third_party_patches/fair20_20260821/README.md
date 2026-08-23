# Fair-20 third-party source patches

These patches freeze the exact dirty source trees used to build the
EfficientQAT, TurboBOA, and YAQA-wclip checkpoints in the 2026-08-21 fair
20-setting campaign. The upstream projects are nested Git repositories and
cannot be represented as ordinary files by the parent `zq` branch, so their
tracked modifications and relevant untracked source/tests/scripts are stored
here explicitly.

| Project | Upstream base commit | Patch | SHA-256 |
|---|---|---|---|
| EfficientQAT | `39f37f3b6053681c9b1cd4c9dcaf692d8999459e` | `efficientqat.patch` | `a403cd7a7a3fdcbba669e9dfecb2ddb31eef2c720a5dcec5d2e9dc3f8e9336e7` |
| TurboBOA | `ea88f93cd4b3730a4731d2dce7bc49971896bb18` | `turboboa.patch` | `0365bcd2857ec09ca96b6737e19af6cdb597c533f16c730120b5aed32be67652` |
| YAQA-wclip | `f9508723251ad839f0162326569f17fe70486fcc` | `yaqa_wclip.patch` | `71e3a9c6e133f52adb95805d454f13cf3f0d478d7b472cf78c0f0b6f4d54fff3` |

Apply each patch from its corresponding upstream checkout after checking out
the exact base commit, for example:

```bash
git checkout 39f37f3b6053681c9b1cd4c9dcaf692d8999459e
git apply /path/to/gptq_plus/third_party_patches/fair20_20260821/efficientqat.patch
```

The patch bundles exclude only generated caches, compiled extensions, and
build products (`__pycache__`, `*.pyc`, `*.so`, and `qtip-kernels/build`).
`git apply --check --reverse` was run against each live campaign source tree
when the bundle was created, proving that applying it to the stated base
reconstructs the current source content.
