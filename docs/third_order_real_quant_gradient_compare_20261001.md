# REAL-Q Fisher vs. matrix-free third-order correction on real W4 FFN residuals

This is an exploratory gradient comparison using the GPTQ+ reference quantization path, not a quantization-quality evaluation. The probe runs immediately after a selected transformer layer has been quantized. Its input is the activation from the actual quantized prefix, its reference is the cached FP output for that layer, and its downstream suffix remains FP. It compares gradients for the target layer's `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj` weights.

## Gradients compared

For target-layer output `y0` on the FP reference path, quantized output `yq`, and actual residual `d = yq - y0`, the output-space gradients are:

* REAL-Q: `gF = F d`, implemented by differentiating the project's `fisher_diag_mse` loss with the collected output Fisher.
* Third-order corrected: `g3 = g0 + gF + 0.5 * (g(y0+d) + g(y0-d) - 2*g0)`, where `g(y) = d KL_downstream / dy` and `g0 = g(y0)`.
* Exact: `g(yq)`, from the full-vocabulary KL through the actual FP suffix to the FP teacher logits.

The symmetric gradient difference estimates the third-derivative contraction with `d` twice without materializing a third-order tensor. It also contains fourth- and higher-order remainder terms. The FP-reference gradient norm is recorded as a check that the teacher/reference paths agree.

Each output-space gradient is mapped to the three FFN weight gradients with the same actual block-output Jacobian, via `torch.autograd.grad(yq, W, grad_outputs=g)`. Reported weight-gradient cosines and relative errors use raw gradients before the optimizer's elementwise gradient clipping.

## Small-run setup

Qwen3-0.6B, W4, GPTQ+ (`enable_gptq_plus=1`), no pre-clip, block size 128, 4 Wikitext-2 calibration sequences of 64 tokens, 2 measured sequences at batch size 1, full-vocabulary KL (`kl_topk=0`). The static empirical Fisher is estimated on the same four calibration sequences; the two measurement sequences are from that pool. Results are therefore a smoke test, not an independent validation set.

## Results

Cosine is against the exact downstream-KL weight gradient. Relative error is `||g_candidate-g_exact|| / ||g_exact||`.

| Layer | FFN projection | Fisher cosine | Fisher relative error | Fisher + cubic cosine | Fisher + cubic relative error |
|---:|---|---:|---:|---:|---:|
| 4 | gate_proj | -0.018 | 2.050 | 0.261 | 1.887 |
| 4 | up_proj | -0.042 | 2.638 | 0.171 | 2.521 |
| 4 | down_proj | -0.001 | 2.597 | 0.221 | 2.490 |
| 20 | gate_proj | 0.028 | 609.658 | 0.029 | 609.669 |
| 20 | up_proj | 0.005 | 597.080 | 0.006 | 597.079 |
| 20 | down_proj | 0.004 | 1017.830 | 0.005 | 1017.839 |

Layer 4's quantized-output residual norm was 0.0622 times the FP output norm; layer 20's was 8.64 times. The FP-reference gradient norms were `1.03e-7` and `1.16e-8`, respectively, consistent with the full-precision teacher/reference KL being effectively zero. Mean measured exact downstream KLs were 0.447 and 1.336.

## Reading the result

At layer 4, the cubic correction rotates the candidate toward the exact gradient and modestly reduces relative error in all three FFN projections. This is an encouraging local signal because the residual is small, but the remaining relative errors are still around 1.9–2.5 and the two measured sequences are not independent of Fisher estimation.

At layer 20, the residual is far outside a small-perturbation regime and the correction barely changes the Fisher gradient. This run cannot fairly test a local Taylor approximation in that regime; it does show that adding this correction does not rescue the severe Fisher mismatch under this tiny calibration setup.

The supported conclusion is narrow: a matrix-free cubic correction is worth replicating at early/mid FFN layers with an independent, larger measurement set and a gradient-clipping-matched readout. It is not yet a working improvement claim. No PPL or final quantization-quality comparison was run.

## Artifacts

* Probe: `tools/run_quantized_third_order_gradient_compare.py`
* Layer 4 metrics: `outputs/third_order_real_quant_layer4/Qwen3-0.6B/third_order_real_quant_layer4/third_order_gradient_compare.json`
* Layer 20 metrics: `outputs/third_order_real_quant_layer20/Qwen3-0.6B/third_order_real_quant_layer20/third_order_gradient_compare.json`
