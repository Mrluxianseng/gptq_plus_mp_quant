from contextlib import contextmanager

from utils import quant_utils, rotation_utils


def configure_activation_quantizers_for_gptq(args, model):
    """Configure the shared A/V runtime sites and return enabled-site counts.

    This is intentionally the single source of truth for legacy aware,
    legacy post-quant (unaware), and refactored REAL-Q paths.  Reconfiguring
    every site, including the fp16 ones, also makes the operation idempotent:
    a model previously configured for A4/V4 cannot retain stale low-bit state
    when reused with A16/V16.
    """

    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    input_count = 0
    v_count = 0
    for name, wrapper in qlayers.items():
        is_lm_head = "lm_head" in name
        is_v_proj = "v_proj" in name
        layer_input_bits = 16 if is_lm_head else args.a_bits

        wrapper.quantizer.configure(
            bits=layer_input_bits,
            groupsize=args.a_groupsize,
            sym=not args.a_asym,
            clip_ratio=args.a_clip_ratio,
        )
        if layer_input_bits < 16:
            input_count += 1

        # V is the only separately quantized projection output.  Explicitly
        # reset every other output site to fp16 so repeated configuration is
        # deterministic and cannot leak an earlier experimental setting.
        layer_output_bits = args.v_bits if is_v_proj else 16
        wrapper.out_quantizer.configure(
            bits=layer_output_bits,
            groupsize=args.v_groupsize,
            sym=not args.v_asym,
            clip_ratio=args.v_clip_ratio,
        )
        if is_v_proj and layer_output_bits < 16:
            v_count += 1

    return input_count, v_count


def configure_k_cache_quantizers_for_gptq(args, analyzer):
    if args.k_bits >= 16:
        return 0
    rope_function_name = "apply_rotary_pos_emb"
    k_quant_config = {
        "k_bits": args.k_bits,
        "k_groupsize": args.k_groupsize,
        "k_sym": not args.k_asym,
        "k_clip_ratio": args.k_clip_ratio,
        "k_quant_enabled": True,
    }
    count = 0
    for layer in analyzer.get_layers():
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_function_name,
            head_dim=analyzer.head_dim,
            **k_quant_config,
        )
        count += 1
    return count


@contextmanager
def disable_fp_path_quant(module):
    act_bits = quant_utils.disable_act_quant(module)
    k_state = rotation_utils.disable_k_cache_quant(module)
    try:
        yield
    finally:
        rotation_utils.enable_k_cache_quant(module, k_state)
        quant_utils.enable_act_quant(module, act_bits)
