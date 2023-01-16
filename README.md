# Fine-tuning SantaCoder on multiple programming languages 🌍
Fine-tune [SantaCoder](https://huggingface.co/bigcode/santacoder) on multiple programming languages for Code Generation using [The Stack](https://huggingface.co/bigcode/the-stack) dataset.

## Fine-tuning SantaCoder

### Setup & Fine-Tuning with The Stack
We provide code to fine-tune the pre-trained [SantaCoder](https://huggingface.co/bigcode/santacoder) model on one of the languages of [The Stack](https://huggingface.co/bigcode/the-stack) dataset. The model has 1B parameters, you can use a local machine with a GPU or train in Google Colab.

You can use the `run_stack.py` script to run the fine-tuning on a local machine, it allows you to launch training using the command line and launch training on multiple GPUs.

For large datasets, we recommend training Santacoder locally instead of in a Google Colab.

1. To begin with, we should clone transformers locally and install all the required packages.

First, you need to clone this repo with:

```
$ git clone https://github.com/bigcode/santacoder-finetuning.git
$ cd santacoder-finetuning
```

Second, install the required packages. The packages are listed in the `requirements.txt` file and can be installed with

```
$ pip install -r requirements.txt
```

2. Next, take a look at the `run_stack.py` script to get an understanding of how it works. In short, the script does the following:

	- Load the given dataset subset
	- Load the model with given hyperparameters
	- Pre-process the dataset to input into the model
	- Run training
	- Run evaluation

3. The following examples show how you can launch fine-tuning for The Stack dataset. 
Here we will run the script on the *Elixir* subset of the dataset for demonstration purposes.


```bash
#!/usr/bin/env bash
python train.py \
--num_train_epochs="30" \
--per_device_train_batch_size="20" \
--per_device_eval_batch_size="20" \
--evaluation_strategy="steps" \
--save_steps="500" \
--eval_steps="100" \
--logging_steps="50" \
--learning_rate="5e-4" \
--warmup_steps="3000" \
--model_name_or_path="bigcode/santacoder" \
--fp16 \
--dataset_name="bigcode/the-stack" \
--dataset_dir="data/elixir" \
--train_split_name="train" \
--validation_split_name="test" \
--preprocessing_num_workers="$(nproc)" \
--verbose_logging \
```

The resulting model and inference examples can be found [here](https://huggingface.co/bigcode/santacoder-elixir).

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
- `vocab.json`
- `config.json`
- `pytorch_model.bin`

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
