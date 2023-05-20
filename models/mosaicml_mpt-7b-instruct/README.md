---
license: cc-by-sa-3.0
datasets:
- mosaicml/dolly_hhrlhf
tags:
- Composer
- MosaicML
- llm-foundry
inference: false
---

# MPT-7B-Instruct

MPT-7B-Instruct is a model for short-form instruction following.
It is built by finetuning [MPT-7B](https://huggingface.co/spaces/mosaicml/mpt-7b) on a [dataset](https://huggingface.co/datasets/sam-mosaic/dolly_hhrlhf) derived from the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and the [Anthropic Helpful and Harmless (HH-RLHF)](https://huggingface.co/datasets/Anthropic/hh-rlhf) datasets. 
  * License: _CC-By-SA-3.0_
  * [Demo on Hugging Face Spaces](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct)


This model was trained by [MosaicML](https://www.mosaicml.com) and follows a modified decoder-only transformer architecture.

## Model Date

May 5, 2023

## Model License

CC-By-SA-3.0

## Documentation

* [Blog post: Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.mosaicml.com/blog/mpt-7b)
* [Codebase (mosaicml/llm-foundry repo)](https://github.com/mosaicml/llm-foundry/)
* Questions: Feel free to contact us via the [MosaicML Community Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-1btms90mc-GipE2ufuPkKY0QBrmF3LSA)!

### Example Question/Instruction

**Longboi24**:
> What is a quoll?

**MPT-7B-Instruct**:

>A Quoll (pronounced “cool”) is one of Australia’s native carnivorous marsupial mammals, which are also known as macropods or wallabies in other parts around Asia and South America

## How to Use

Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. This is because we use a custom model architecture that is not yet part of the `transformers` package.

It includes options for many training efficiency features such as [FlashAttention (Dao et al. 2022)](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi](https://arxiv.org/abs/2108.12409), QK LayerNorm, and more.

```python
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  trust_remote_code=True
)
```
Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. 
This is because we use a custom `MPT` model architecture that is not yet part of the Hugging Face `transformers` package.
`MPT` includes options for many training efficiency features such as [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi](https://arxiv.org/abs/2108.12409), [QK LayerNorm](https://arxiv.org/abs/2010.04245), and more.

To use the optimized [triton implementation](https://github.com/openai/triton) of FlashAttention, you can load the model with `attn_impl='triton'` and move the model to `bfloat16`:
```python
config = transformers.AutoConfig.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  trust_remote_code=True
)
config.attn_config['attn_impl'] = 'triton'

model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True
)
model.to(device='cuda:0')
```

Although the model was trained with a sequence length of 2048, ALiBi enables users to increase the maximum sequence length during finetuning and/or inference. For example:

```python
config = transformers.AutoConfig.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  trust_remote_code=True
)
config.update({"max_seq_len": 4096})
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  config=config,
  trust_remote_code=True
)
```

This model was trained with the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
```

## Model Description

The architecture is a modification of a standard decoder-only transformer.

The model has been modified from a standard transformer in the following ways:
* It uses [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf)
* It uses [ALiBi (Attention with Linear Biases)](https://arxiv.org/abs/2108.12409) and does not use positional embeddings
* It does not use biases


| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 6.7B |
|n_layers | 32 |
| n_heads | 32 |
| d_model | 4096 |
| vocab size | 50432 |
| sequence length | 2048 |

## PreTraining Data

For more details on the pretraining process, see [MPT-7B](https://huggingface.co/mosaicml/mpt-7b).

The data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

## Limitations and Biases

_The following language is modified from [EleutherAI's GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)_

MPT-7B-Instruct can produce factually incorrect output, and should not be relied on to produce factually accurate information.
MPT-7B-Instruct was trained on various public datasets.
While great efforts have been taken to clean the pretraining data, it is possible that this model could generate lewd, biased or otherwise offensive outputs.


## Acknowledgements

This model was finetuned by Sam Havens and the MosaicML NLP team

## MosaicML Platform

If you're interested in [training](https://www.mosaicml.com/training) and [deploying](https://www.mosaicml.com/inference) your own MPT or LLMs on the MosaicML Platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=huggingface&utm_medium=referral&utm_campaign=mpt-7b).

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please cosult an attorney before using this model for commercial purposes.

## Citation

Please cite this model using the following format:

```
@online{MosaicML2023Introducing,
    author    = {MosaicML NLP Team},
    title     = {Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs},
    year      = {2023},
    url       = {www.mosaicml.com/blog/mpt-7b},
    note      = {Accessed: 2023-03-28}, % change this date
    urldate   = {2023-03-28} % change this date
}
```