from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_token_nll(model, dataloader, pad_id):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    device=model.device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Model inferencing", total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits[:, :-1, :]
            targets = batch["input_ids"][:, 1:]

            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            mask = targets != pad_id

            nll = F.cross_entropy(
                logits[mask],
                targets[mask],
                reduction="sum"
            )

            total_nll += nll.item()
            total_tokens += mask.sum().item()

    return total_nll / total_tokens

if __name__ == "__main__":
    from functools import partial
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset("parquet", data_files=args.dataset, split="train")

    def collate_fn(sequences, pad_id):
        max_len = max(len(x["input_ids"]) for x in sequences)

        input_ids = [i["input_ids"] for i in sequences]
        attention_mask = [i["attention_mask"] for i in sequences]
        labels = [i["labels"] for i in sequences]

        for i in range(len(input_ids)):
            pad_len = max_len - len(input_ids[i])

            input_ids[i] = torch.tensor([pad_id] * pad_len + input_ids[i])
            attention_mask[i] = torch.tensor([0] * pad_len + attention_mask[i])
            labels[i] = torch.tensor([-100] * pad_len + labels[i])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    _collate_fn = partial(collate_fn, pad_id=tokenizer.pad_token_id)

    dataloader = DataLoader(
        dataset, # type: ignore
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=_collate_fn
    )

    nll = compute_token_nll(
        model,
        dataloader,
        pad_id=tokenizer.pad_token_id
    )

    print(f"NLL: {nll}")
