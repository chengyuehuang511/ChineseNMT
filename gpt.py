from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import AdamW
import utils
import config
from train import evaluate1
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
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 15
    max_seq_length = 20
    lossl = []
    b = []
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):  # Assuming data_loader is set up to yield input and labels
            attention_mask = batch.src
            attention_mask[attention_mask!=0] = 1
            outputs = model(input_ids=batch.src[:,1:], labels=batch.trg[:,1:],attention_mask=attention_mask[:,1:])
            loss = outputs.loss
            lossl.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print("-----")
        bleu_score = evaluate1(dev_dataloader, model, device)
        print(epoch, bleu_score)
        logging.info('Epoch: {}, Bleu Score: {}'.format(epoch, bleu_score))
        b.append(bleu_score)
    print(b)
    
run()