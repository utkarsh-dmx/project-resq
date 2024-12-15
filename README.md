
# ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals
This repository contains the code for the submitted paper.



## Abstract
Post-training quantization (PTQ) of large language models (LLMs) holds the promise in reducing the prohibitive computational cost at inference time. Quantization of all weight, activation and key-value (KV) cache tensors to 4-bit without significantly degrading generalizability is challenging, due to the high quantization error caused by extreme outliers in activations. To tackle this problem, here we propose **ResQ**, a PTQ method that pushes further the state-of-the-art. By means of principal component analysis (PCA), it identifies a low-rank subspace (in practice 1/8 of the hidden dimension) in which activation variances are highest, and keep the coefficients within this subspace in high precision, e.g.~8-bit, while quantizing the rest to 4-bit. Within each subspace, invariant random rotation is applied to further suppress outliers.  We show that this is a provably optimal mixed precision quantization scheme that minimizes error. With the Llama families of models, we demonstrate that ResQ outperforms recent uniform and mixed precision PTQ methods on a variety of benchmarks, achieving up to 33\% lower perplexity on Wikitext than the next best method SpinQuant, at a 2.4x speedup over 16-bit baseline.


## Usage
Tested with Python=3.9.19, CUDA=11.8, pytorch=2.3.0 on NVIDIA A100
1. conda create -n "resq" python=3.9.19
2. Install pytorch with CUDA from [here](https://pytorch.org/get-started/locally/) according to your system specifications. The code below should also work. 
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
3. Install ResQ requirements
```
pip install -r requirements.txt
```
4. Install fast-hadamard-transform from [here](https://github.com/Dao-AILab/fast-hadamard-transform)
```
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
```


## Running the code
We have given step by step instructions to evaluate Llama-3.2-3b on MMLU and Wiktext.
1. Get ResQ projections by running
```
bash 0_get_basis.sh
```
2. Quantize and evaluate the model using
```
bash 2_eval_ptq.sh
```


