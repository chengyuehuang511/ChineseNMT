from transformers import BertModel, BertTokenizer, AdamW
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import utils
import config
from train import evaluate1  # Make sure this is adapted for BERT
from data_loader import MTDataset

class BERTForNMT(nn.Module):
    def __init__(self, bert_model_name='bert-base-multilingual-cased', num_labels=None):
        super(BERTForNMT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.num_labels = num_labels  # This should be the size of the target vocabulary
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return logits, loss
        return logits



def run():
    utils.set_logger(config.log_path)
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build!--------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, collate_fn=test_dataset.collate_fn)
    logging.info("-------- Get Dataloader!--------")
    
    model = BertForNMT(70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            model.train()
            src_input_ids, tgt_input_ids = batch.src.to(device), batch.trg.to(device)
            attention_mask = (src_input_ids != 0).long().to(device)  # Create attention mask properly
            decoder_logits = model(src_input_ids, attention_mask, tgt_input_ids[:, 0], device)
            logits_flat = decoder_logits.view(-1, decoder_logits.size(-1))
            targets_flat = tgt_input_ids[:, 1:].contiguous().view(-1)  # Flatten targets to match logits

        # Compute the loss
            loss = criterion(logits_flat, targets_flat) 

        # Perform backward pass to compute gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logging.info("Epoch: {}, loss: {}".format(epoch, loss.item()))
            model.eval()
            print("-----")
            bleu_score = evaluate1(dev_dataloader, model, device)  # Ensure evaluate1 is correctly defined and called
            print(f"Epoch {epoch}, Loss: {loss.item()}, BLEU Score: {bleu_score}")
run()