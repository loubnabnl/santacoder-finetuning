"""
Fine-Tune SantaCoder on code/text dataset
"""

import argparse
import os
import numpy as np
import functools
import timeit

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/santacoder")
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-dedup")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--data_column", type=str, default="content")


    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    parser.add_argument("--fim_rate", type=float, default=0.5)
    parser.add_argument("--fim_spm_rate", type=float, default=0.5)
    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens

# this is expensive so we cache it
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    _, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map['additional_special_tokens']
    suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD])
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(sample, np_rng, tokenizer, fim_rate=0.5, fim_spm_rate=0.5, truncate_or_pad=False):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it. 
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id, = get_fim_token_ids(tokenizer)

    if np_rng.binomial(1, fim_rate): # sample bernoulli dist
        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(sample), sample)
            print(e)
            raise e

        prefix = np.array(sample[:boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0]:boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1]:], dtype=np.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)
            if diff > 0: # too long
                if suffix.shape[0] <= diff: # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample, np_rng
                suffix = suffix[:suffix.shape[0] - diff]
            elif diff < 0: # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])
        
        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate([
                [prefix_tok_id, suffix_tok_id], suffix,
                [middle_tok_id], prefix, middle
            ])
        else:
            # PSM
            new_sample = np.concatenate([
                [prefix_tok_id], prefix,
                [suffix_tok_id], suffix,
                [middle_tok_id], middle
            ])
    else:
        # don't do FIM preproc
        new_sample = sample

    return list(new_sample), np_rng

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else args.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            np_rng = np.random.RandomState(seed=555)
            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(tokenized_input, np_rng, self.tokenizer, fim_rate=self.fim_rate, fim_spm_rate=self.fim_spm_rate, truncate_or_pad=False)
                
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate
    )

    valid_dataset_no_fim = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=0.0,
        fim_spm_rate=0.0
    )
    return train_dataset, valid_dataset, valid_dataset_no_fim


def run_training(args, train_data, val_data):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_cache=not args.no_gradient_checkpointing,
    )
    train_data.start_iteration = 0

    print(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        fp16=args.no_fp16,
        weight_decay=args.weight_decay,
        run_name=f"santacoder-{args.subset}",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    train_dataset, eval_dataset, eval_no_fim = create_datasets(tokenizer, args)

    fim_time = timeit.timeit(lambda: next(iter(eval_dataset)), number=5)
    non_fim_time = timeit.timeit(lambda: next(iter(eval_no_fim)), number=5)

    print(f"FIM time: {fim_time}")
    print(f"Non-FIM time: {non_fim_time}")

    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
