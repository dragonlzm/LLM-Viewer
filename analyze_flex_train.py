from flexible_analyzer import FlexibleAnalyzer
import torch.nn as nn
import numpy as np
import os
import importlib
import argparse
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument(
    "hardware",
    type=str,
    help="name of hardware, for example nvidia_V100 or nvidia_A6000",
)
parser.add_argument("--config_file", type=str, default=None, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
parser.add_argument("--promptlen", type=int, default=128, help="prompt sequence length")
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
parser.add_argument("--source", dest="source",  type=str, default="huggingface", help="use flash attention")
parser.add_argument(
    "--tp-size",
    type=int,
    default=1,
    help="the number of devices for tensor parallelism to use"
)
args = parser.parse_args()

analyzer = FlexibleAnalyzer(args.model_id, args.hardware, args.config_file,source=args.source)

number_of_layer_of_model = analyzer.config.get_num_hidden_layers(analyzer.model_params)
num_attention_heads = analyzer.config.get_num_attention_heads(analyzer.model_params)

ipdb.set_trace()
results = analyzer.analyze_allq_layers(
    prompt_len=[args.promptlen]*number_of_layer_of_model,
    num_heads=[num_attention_heads]*number_of_layer_of_model,
    batchsize=args.batchsize,
    w_bit=args.w_bit,
    a_bit=args.a_bit,
    kv_bit=args.kv_bit,
    use_flashattention=args.use_flashattention,
    tp_size=args.tp_size
)
print(results)