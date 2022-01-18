import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    dataset = load_dataset('glue', 'mrpc', split='train')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    print(dataset[:3])
    dataset = dataset.map(lambda e: tokenizer(e['sentence1'], truncation=True, padding='max_length'), batched=True)
    print(dataset[:3])
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataloader = DataLoader(dataset, batch_size=32)
    sample = next(iter(dataloader))
    # print(sample)



if __name__ == "__main__":
    main()