# Fine-tuning SantaCoder on multiple programming languages 🌍
Fine-tune [SantaCoder](https://huggingface.co/bigcode/santacoder) on multiple programming languages for Code Generation using [The Stack](https://huggingface.co/bigcode/the-stack) dataset. SantaCoder is 1B parameters model pre-trained on Python, Java & JavaScript, we suggest fine-tuning on languages close to them, otherwise the model might not converge well.


## Setup & Fine-Tuning with The Stack

We provide code to fine-tune the pre-trained [SantaCoder](https://huggingface.co/bigcode/santacoder) model on one of the languages of [The Stack](https://huggingface.co/bigcode/the-stack) dataset (after [near-deduplication](https://huggingface.co/datasets/bigcode/the-stack-dedup)). The code can be adapted to fine-tune on other code datasets. Check this [repository](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning) for fine-tuning models on some code tasks. You can also find other resources in [CodeParrot repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot), such as training code models with `accelerate` and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

You can use the `run_stack.py` script to run the fine-tuning on a local machine, it allows you to launch training using the command line and launch training on multiple GPUs.

1. To begin with, we should clone the repository locally, install all the required packages and log to HuggingFace Hub and Weight & Biases.

First, you can clone this repo with:

```
git clone https://github.com/bigcode/santacoder-finetuning.git
cd santacoder-finetuning
```

Second, install the required packages. The packages are listed in the `requirements.txt` file and can be installed with

```
pip install -r requirements.txt
```

Third, make sure you are logged to HuggingFace Hub and Weights & Biases

```
huggingface-cli login
wandb login
```

2. Next, take a look at the `run_stack.py` script to get an understanding of how it works. In short, the script does the following:

	- Load the given dataset subset
	- Load the model with given hyperparameters
	- Pre-process the dataset to input into the model
	- Run training
	- Run evaluation

3. The following examples show how you can launch fine-tuning for The Stack dataset. 
Here we will run the script on the *Ruby* subset of the dataset for demonstration purposes. Note that:
- Gradient Checkpointing are enabled by default and the caching mechanism is disabled to save memory. If you want to disable them call `no_gradient_checkpointing` argument. Note that Mixed precision is disabled with the `no_fp16` flag due to some issues we noticed when using it, you can enable it by removing that argument.
- If the model still doesn't fit in your memory use `batch_size` 1 and reduce `seq_length` to 1024 for example.
- If you want to use [streaming](https://huggingface.co/docs/datasets/stream) and avoid downloading the entire dataset, add the flag `streaming`.


```bash
#!/usr/bin/env bash
python run_stack.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="bigcode/the-stack-dedup" \
        --subset="data/ruby" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 500 \
        --eval_freq 3000 \
        --save_freq 3000 \
        --log_freq 1 \
        --num_workers="$(nproc)" \
	--no_fp16
```

The resulting model and inference examples can be found [here](https://huggingface.co/bigcode/santacoder-ruby).

## How to upload my trained checkpoint

To upload your trained checkpoint, you have to create a new model repository on the 🤗 model hub, from this page: https://huggingface.co/new

> You can also follow the more in-depth instructions [here](https://huggingface.co/transformers/model_sharing.html) if needed.

Having created your model repository on the hub, you should clone it locally:

```bash
git lfs install

git clone https://huggingface.co/username/your-model-name
```

Then and add the following files that fully define a SantaCoder checkpoint into the repository. You should have added the following files.

- `tokenizer_config.json`
- `tokenizer.json`
- `config.json`
- `pytorch_model.bin`
- modleing files (see below)

You can get the tokenizer files by cloning the [model repo](https://huggingface.co/bigcode/santacoder/tree/main). Santacoder currently has a custom [modeling file](https://huggingface.co/bigcode/santacoder/blob/main/modeling_gpt2_mq.py) + config file on the hub, but they will be included with the saved checkpoints if you used the `transformers` branch in `requirements.txt`.

Having added the above files, you should run the following to push files to your model repository.  
```
git add . && git commit -m "Add model files" && git push
```

The next **important** step is to create the model card. For people to use your fine-tuned 
model it is important to understand: 

- What kind of model is it?
- What is your model useful for?
- What data was your model trained on?
- How well does your model perform?

All these questions should be answered in a model card which is the first thing people see when 
visiting your model on the hub under `https://huggingface.co/{your_username}/{your_modelname}`.

## Acknowledgments

This is inspired by the [Wave2vec fine-tuning week](https://github.com/huggingface/transformers/edit/main/examples/research_projects/wav2vec2/) by [Hugging Face](https://huggingface.co/).
