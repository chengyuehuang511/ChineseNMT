from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import AdamW
import utils
import config
from train import evaluate
from data_loader import MTDataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

def run():
    utils.set_logger(config.log_path)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build!--------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    logging.info("-------- Get Dataloader!--------")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 40
    max_seq_length = 70
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):  # Assuming data_loader is set up to yield input and labels
            inputs = tokenizer(batch.src_text, return_tensors='pt', padding=True,
             truncation=True,max_length=max_seq_length).to(device)
            labels = tokenizer(batch.trg_text, return_tensors='pt', padding=True,
             truncation=True,max_length=max_seq_length)['input_ids'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print("-----")
        bleu_score = evaluate(dev_dataloader, model)
        print(epoch, bleu_score)
        logging.info('Epoch: {}, Bleu Score: {}'.format(epoch, bleu_score))
    
run()