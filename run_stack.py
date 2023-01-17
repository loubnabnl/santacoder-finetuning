"""
Fine-Tune SantaCoder on a new language from The Stack
"""

import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
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
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack")
    parser.add_argument("--subset", type=str, default="data/python")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=250, type=int)
    parser.add_argument("--save_freq", default=250, type=int)
    return parser.parse_args()


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
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = "content"

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
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }


def create_dataloaders(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers,
    )
    dataset = dataset.train_test_split(test_size=0.005, shuffle=False, seed=args.seed)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    train_data = train_data.shuffle(seed=args.seed)
    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    train_data.start_iteration = 0

    print(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        num_train_epochs=10,
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
        weight_decay=args.weight_decay,
        fp16=args.fp16,
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
    train_dataset, eval_dataset = create_dataloaders(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
