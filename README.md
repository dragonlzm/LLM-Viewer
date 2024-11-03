# LLM-Viewer

<img src="figs/eye.png" alt="LLM-Viewer" width="50"/>


LLM-Viewer is a tool for visualizing Language and Learning Models (LLMs) and analyzing the performance on different hardware platforms. It enables network-wise analysis, considering factors such as peak memory consumption and total inference time cost. With LLM-Viewer, you can gain valuable insights into LLM inference and performance optimization.
You can use LLM-Viewer in a web browser or as a command line interface (CLI) tool. The web version provides a user-friendly interface for easy configuration and visualization, you can access it at [LLM-Viewer Web](http://llm-viewer.com).

We invite you to read our paper [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363.pdf).
In this paper, we provide a comprehensive analysis of the latest advancements in efficient LLM inference using LLM-Viewer. 

This ongoing project will be updated. TODO list:
- Show shape of tensors.
- Pre-process and post-process for non-transformer layers.
- Show the whole network.
- Expand hardware platform compatibility and allow manual configuration of hardware parameters.
- Increase support for more LLMs and enable manual configuration of model graphs.

## Workflow

![LLM-Viewer Workflow](figs/workflow.svg)

As shown in the Figure, the workflow consists of the following steps:

1. Input the LLM and gather essential information about each layer, including the computation count, input and output tensor shapes, and data dependencies.
2. Provide input for the hardware and generate a roofline model that takes into account the computation capacity and memory bandwidth of the hardware.
3. Configure the inference settings, such as the batch size, prompt token length, and generation token length.
4. Configure the optimization settings, such as the quantization bitwidth, utilization of FlashAttention, decoding methods, and other system optimization techniques.
5. Use the LLM-Viewer Analyzer to analyze the performance of each layer based on the roofline model and layer information. It also tracks the memory usage of each layer and calculates the peak memory consumption based on data dependencies. The overall network performance of the LLM can be obtained by aggregating the results of all layers.
6. Generate a report that provides information such as the maximum performance and performance bottlenecks of each layer and the network, as well as the memory footprint. The report can be used to analyze curves, such as batch size-performance and sequence length-performance curves, to understand how different settings impact performance.
7. Access the LLM-Viewer web viewer for convenient visualization of the network architecture and analysis results. This tool facilitates easy configuration adjustment and provides access to various data for each layer.

## Web Usage

To use LLM-Viewer in a web browser, go to the web-site [LLM-Viewer Web](http://llm-viewer.com).
You can click the node to get the detailed analysis of the layer.

## CLI Usage

Clone the LLM-Viewer repository from GitHub: 
```git clone https://github.com/hahnyuan/LLM-Viewer.git   ```

Install requirements
```pip install transformers flask flask_cors easydict```

To analyze an LLM using LLM-Viewer in command line interface (cli), run the following command:

```bash
python3 analyze_cli.py facebook/opt-125m nvidia_A6000
python3 analyze_cli.py meta-llama/Llama-2-7b-hf nvidia_A6000 --batchsize 1 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 16 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 8192

# DiT models
python3 analyze_cli.py DiT-XL/2 nvidia_A6000 --batchsize 1 --seqlen 256 --source DiT
```

NOTE: The time estimated by the roofline model represents the theoretical performance that the hardware can achieve. 
The purpose of creating this tool is to help readers gain a clearer understanding of the key factors that influence LLM inference. 
Only the relative relationships can be referenced. 

## Citation

If you are using LLM-Viewer in your research, please cite our paper:

```
@misc{yuan2024llm,
      title={LLM Inference Unveiled: Survey and Roofline Model Insights}, 
      author={Zhihang Yuan and Yuzhang Shang and Yang Zhou and Zhen Dong and Chenhao Xue and Bingzhe Wu and Zhikai Li and Qingyi Gu and Yong Jae Lee and Yan Yan and Beidi Chen and Guangyu Sun and Kurt Keutzer},
      year={2024},
      eprint={2402.16363},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## How to use this script to eval the llava-onevision
```
# To calculate the prefill only
python3 analyze_flex_prefill_only.py lmms-lab/llava-onevision-qwen2-0.5b-ov nvidia_A100 --config_file configs/Llama.py --promptlen 1024

python3 analyze_flex_prefill_only.py lmsys/vicuna-7b-v1.5 nvidia_A100 --config_file configs/Llama.py  --promptlen 616

# To calculate the prefill with different token numbers at different layers
python3 analyze_flex_prefill_only.py lmms-lab/llava-onevision-qwen2-0.5b-ov nvidia_A100 --config_file configs/Llama.py --promptlen 1024,1024,1024,1024,512,512,512,512,256,256,256,256,128,128,128,128,64,64,64,64,32,32,32,32

# To calculate the prefill and the decode
python3 analyze_flex.py lmms-lab/llava-onevision-qwen2-0.5b-ov nvidia_A100 --config_file configs/Llama.py --promptlen 1024,1024,1024,1024,512,512,512,512,256,256,256,256,128,128,128,128,64,64,64,64,32,32,32,32 --seqlen 10

# Other examples
python3 analyze_cli.py lmsys/vicuna-7b-v1.5 nvidia_V100 --config_file configs/Llama.py --seqlen 616

python3 analyze_flex_train.py lmsys/vicuna-7b-v1.5 nvidia_V100 --config_file configs/Llama.py --promptlen 616

python3 analyze_gen_cli.py lmsys/vicuna-7b-v1.5 nvidia_V100 --config_file configs/Llama.py --promptlen 616 --seqlen 1

python3 analyze_flex_train.py lmsys/vicuna-7b-v1.5 nvidia_V100 --config_file configs/Llama.py --promptlen 576 --seqlen 40


# calculate the data for the training
python3 analyze_flex_train.py lmms-lab/llava-onevision-qwen2-0.5b-ov nvidia_A6000 --config_file configs/Llama.py --promptlen 2048

python3 analyze_flex_train.py lmms-lab/llava-onevision-qwen2-0.5b-ov nvidia_A100 --config_file configs/Llama.py --promptlen 2048


```

##  How to read the number from the result 
The sample of the result output is as following:
{'decode': {'OPs': 1031776256, 'memory_access': 1020557824.0, 'load_weight': 987922432.0, 'load_act': 3546880.0, 'store_act': 3910400.0, 'load_kv_cache': 25165824.0, 'store_kv_cache': 12288.0, 'inference_time': 0.0006563072823151125, 'memory_consumption': 740969216.0, 'memory_consumption_tmp_act': 150272.0, 'memory_consumption_weight': 715653120.0, 'memory_consumption_kv_cache': 25165824.0}, 

'prefill': {'OPs': 1834410131456, 'memory_access': 15609571840.0, 'load_weight': 987922432.0, 'load_act': 7184844544.0, 'store_act': 7386473216.0, 'load_kv_cache': 25165824.0, 'store_kv_cache': 25165824.0, 'inference_time': 0.012891532104373953, 'memory_consumption': 1048576000.0, 'memory_consumption_tmp_act': 307757056.0, 'memory_consumption_weight': 715653120.0, 'memory_consumption_kv_cache': 25165824.0}}

The results consist of two parts: the 'decode' section and the 'prefill' section.
the 'decode' section is calculate the resource need for decoding one token after prefiling stage. While the 'prefill' calculate the flop in the prefiling stage. Usually, the existing model will use the 