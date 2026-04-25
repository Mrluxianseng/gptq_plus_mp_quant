# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import os
import logging

import torch
import transformers
from accelerate.hooks import remove_hook_from_module

from gptq_utils import gptq_utils, gptaq_utils, gptq_guided_utils, gptq_plus_utils
from utils import data_utils, model_utils


def quantize_weights(args, analyzer: model_utils.ModelAnalyzer):
    transformers.set_seed(args.seed)

    model = analyzer.model
    if not bool(getattr(model, "_gptqplus_fsdp_prepared", False)):
        model.cpu()
    remove_hook_from_module(model, recurse=True)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            logging.info("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        else:
            trainloader = data_utils.get_tokens(args.dataset, "train", analyzer.tokenizer, args.seq_len,
                                                args.nsamples, args.tokens_cache_path, args.seed)
            
            if isinstance(trainloader[0], torch.Tensor):
                assert trainloader[0].dim() == 1
                logging.info("Reformatting input tokens to tuple + Unsqueeze")
                trainloader = [(x.unsqueeze(0), None) for x in trainloader]

            # DP-aware device selection: with torchrun each rank calls
            # `torch.cuda.set_device(LOCAL_RANK)` before entering this function,
            # so `cuda:<current>` points at that rank's physical GPU. Hard-coding
            # "cuda:0" here would send rank>0's tensors to physical GPU 0 while
            # NCCL talks on the current device, which deadlocks the collective.
            if torch.cuda.is_available():
                dp_dev = f"cuda:{torch.cuda.current_device()}"
            else:
                dp_dev = "cpu"

            if args.w_method == "rtn":
                quantizers = gptq_utils.rtn_fwrd(args, analyzer, dp_dev)
            elif args.w_method == "gptq":
                quantizers = gptq_utils.gptq_fwrd(args, analyzer, trainloader, dp_dev)
            elif args.w_method == "gptaq":
                quantizers = gptaq_utils.gptq_fwrd(args, analyzer, trainloader, dp_dev)
            elif args.w_method == "gptq_guided":
                quantizers = gptq_guided_utils.gptq_fwrd(args, analyzer, trainloader, dp_dev)
            elif args.w_method == "gptq_plus":
                quantizers = gptq_plus_utils.gptq_fwrd(args, analyzer, trainloader, dp_dev)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            os.makedirs(os.path.dirname(args.save_qmodel_path), exist_ok=True)
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    return model
