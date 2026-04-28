from contextlib import contextmanager

from utils import quant_utils, rotation_utils


def configure_activation_quantizers_for_gptq(args, model):
    """Enable ActQuantWrapper fake quant for weight-quantization student paths."""
    if args.a_bits >= 16 and args.v_bits >= 16:
        return 0, 0

    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    input_count = 0
    v_count = 0
    for name, wrapper in qlayers.items():
        layer_input_bits = args.a_bits
        if "lm_head" in name:
            layer_input_bits = 16

        wrapper.quantizer.configure(
            bits=layer_input_bits,
            groupsize=args.a_groupsize,
            sym=not args.a_asym,
            clip_ratio=args.a_clip_ratio,
        )
        if layer_input_bits < 16:
            input_count += 1

        if "v_proj" in name and args.v_bits < 16:
            wrapper.out_quantizer.configure(
                bits=args.v_bits,
                groupsize=args.v_groupsize,
                sym=not args.v_asym,
                clip_ratio=args.v_clip_ratio,
            )
            v_count += 1

    return input_count, v_count


def configure_k_cache_quantizers_for_gptq(args, analyzer):
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
