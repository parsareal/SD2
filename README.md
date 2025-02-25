<div align="center">
  <h2><i>S2D:</i> Sorted Speculative Decoding For More Efficient <br> Deployment of
Nested Large Language Models</h2> 
</div>
<p align="center">
 <a href="https://www.overleaf.com/project/663258f7b74d2ee3df67880a"><b>Paper</b></a> 
<!-- | <a href="https://sites.google.com/view/spec-bench/"><b>Blog</b></a> | <a href="https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md"><b>Leaderboard</b></a> | <a href="ROADMAP.md"><b>Roadmap</b></a> | -->
</p>





![timeline](./inference/assets/methodology.png)

<div align="center">
<font color="gray">Speedup comparison of Speculative Decoding methods on Spec-Bench.</font>
</div>

## Introduction

Sorted Speculative Decoding  (S2D) is a method providing the capability of selecting multiple draft models adaptively based on the given target. Without the need for training separate draft models for different target models, S2D enjoys the flexibility of having different submodels in the same architecture, which causes the approach outperforms other baselines in multi-target speculative decoding scenario.
<!-- Spec-Bench is a comprehensive benchmark designed for assessing Speculative Decoding methods across diverse scenarios. Based on Spec-Bench, we aim to establish and maintain a unified evaluation platform for open-source Speculative Decoding approaches. This platform facilitates the systematic assessment of existing methods ***in the same device and testing environment***, thereby ensuring fair comparisons.  -->

# Train

We used the codebase of [fastchat](https://github.com/lm-sys/FastChat/tree/main) to train our draft models on ShareGPT dataset.

# Instalation
```
conda create -n sd2train python=3.9
conda activate sd2train
cd train
pip install -r requirements.txt
```

# Finetuning
To train the draft model, you need to first change the ```num_hidden_layers``` attribute to 12 in the ```config.json``` of the Vicuna 7b pre-trained checkpoint path. You can set this by runing ```vi {Vicuna7b_path}/config.json```. 
After changing the config to only pick the first 12 layers, you can run the training in either SFT or SoFT.
```
cd train
```
1) training SFT model
```
sh scripts/train_draft_sft.sh
```
2) training SoFT model
```
sh scripts/train_draft_soft.sh
```
## Model Weights

Download corresponding model weights (if required) and modify the checkpoint path in `eval.sh`.

- [vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [vicuna-13B-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- [SD2-SoFT-Draft](https://huggingface.co/parsakaveh/SD2-SoFT-Draft)

<!-- - [EAGLE](https://github.com/SafeAILab/EAGLE?tab=readme-ov-file#eagle-weights)
- [Hydra](https://github.com/zankner/hydra?tab=readme-ov-file#model-weights)
- [Medusa-1](https://github.com/FasterDecoding/Medusa?tab=readme-ov-file#medusa-1)
- [Speculative Sampling](https://github.com/NJUNLP/MCSD?tab=readme-ov-file#model-release) -->

<!-- ## Additonal Setup -->

<!-- #### REST (Optional) -->

<!-- ##### Build DraftRetriever from source -->

<!-- ```
cd model/rest/DraftRetriever
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
maturin build --release --strip -i python3.9 # will produce a .whl file
pip3 install ./target/wheels/draftretriever-0.1.0-cp39-cp39-linux_x86_64.whl
``` -->
<!-- 
##### Create a datastore

```
cd model/rest/datastore
./datastore.sh # modify your own path
``` -->



# Evaluation

We used the [Spec-Bench](https://github.com/hemingkx/Spec-Bench/tree/main) code to evaluate our S2D approach and other baselines on multiple domains.

## Installation

```
conda create -n sd2eval python=3.10
conda activate sd2eval
cd inference
pip install -r requirements.txt
```


Currently, Spec-Bench supports the evaluation of the following open source models:

- [EAGLE](https://sites.google.com/view/eagle-llm)
- [Hydra](https://github.com/zankner/hydra)
- [Medusa](https://sites.google.com/view/medusa-llm)
- [Speculative Sampling](https://huggingface.co/blog/assisted-generation)
- [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
- [REST](https://sites.google.com/view/rest-llm/)
- [Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)

## Inference

Select specific command line in `eval.sh`, the results will be stored in `data/spec_bench/model_answer/`.

```
cd inference
./eval.sh
```

## Speedup Report

Obtain the corresponding speedup compared to vanilla autoregressive decoding.

```
python evaluation/speed.py --file-path /your_own_path/s2d.jsonl --base-path /your_own_path/vicuna.jsonl
```

## Result Comparison

Examine whether the generated results are equal to autoregressive decoding or not.

```
python evaluation/equal.py --file-path /your_own_path/model_answer/ --jsonfile1 vicuna.jsonl --jsonfile2 s2d.jsonl
```
<!-- 
## Contributing

We warmly welcome contributions and discussions related to Spec-Bench! If you have any suggestions for improvements or ideas you'd like to discuss, please don't hesitate to open an issue. This will allow us to collaborate and discuss your ideas in detail.

***More models are welcome!*** - If you're aware of any open-source Speculative Decoding methods not currently included in Spec-Bench, we encourage you to contribute by submitting a pull request. This helps ensure Spec-Bench remains a comprehensive and fair benchmarking platform for comparing existing methods. Please ensure that your changes are well-tested before submission. -->

<!-- ## Acknowledgments

This codebase is built from [Medusa](https://github.com/FasterDecoding/Medusa) and [EAGLE](https://github.com/SafeAILab/EAGLE). We integrated code implementations of multiple open-source Speculative Decoding methods to facilitate unified evaluation. -->

<!-- ## Citation

If you find the resources in this repository useful, please cite our paper:

```
@misc{xia2024unlocking,
      title={Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding}, 
      author={Heming Xia and Zhe Yang and Qingxiu Dong and Peiyi Wang and Yongqi Li and Tao Ge and Tianyu Liu and Wenjie Li and Zhifang Sui},
      year={2024},
      eprint={2401.07851},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``` -->

