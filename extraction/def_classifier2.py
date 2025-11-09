# Input / Output
source_folder = 'def_data'
destination_folder = 'def_data/Model'

# Libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from sklearn.metrics import classification_report
import pandas as pd

# Prepare
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SEQ_LEN = 128

# Dataset
class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Fill missing values
        self.df.fillna('', inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Merge title + text
        titletext = self.df.iloc[idx]['titletext'] if 'titletext' in self.df.columns else ''
        text_ids = tokenizer.encode(
            titletext,
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding='max_length'
        )
        label = self.df.iloc[idx]['label']
        return {
            'input_ids': torch.tensor(text_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Datasets
train_dataset = TextDataset(os.path.join(source_folder, 'train.csv'))
valid_dataset = TextDataset(os.path.join(source_folder, 'valid.csv'))
test_dataset  = TextDataset(os.path.join(source_folder, 'test.csv'))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# BERT Module
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, labels=None):
        output = self.encoder(input_ids=input_ids, labels=labels)
        loss = output.loss
        logits = output.logits
        return loss, logits

# Save/Load functions (unchanged)
def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return
    torch.save({'model_state_dict': model.state_dict(), 'valid_loss': valid_loss}, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    print(f'Model loaded from <== {load_path}')
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return
    torch.save({
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list,
        'global_steps_list': global_steps_list
    }, save_path)
    print(f'Metrics saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Metrics loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training function
def train(model, optimizer, train_loader, valid_loader, num_epochs=10, eval_every=None, file_path=destination_folder):
    if eval_every is None:
        eval_every = len(train_loader) // 2

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list, valid_loss_list, global_steps_list = [], [], []
    best_valid_loss = float("inf")

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # Evaluation
            if global_step % eval_every == 0:
                model.eval()
                valid_running_loss = 0.0
                with torch.no_grad():
                    for batch in valid_loader:
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['label'].to(device)
                        loss, _ = model(input_ids, labels)
                        valid_running_loss += loss.item()
                avg_train_loss = running_loss / eval_every
                avg_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(avg_train_loss)
                valid_loss_list.append(avg_valid_loss)
                global_steps_list.append(global_step)

                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}], '
                      f'Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')

                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    save_checkpoint(os.path.join(file_path, 'model.pt'), model, best_valid_loss)
                    save_metrics(os.path.join(file_path, 'metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)

                running_loss = 0.0
                model.train()

    save_metrics(os.path.join(file_path, 'metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

# Evaluation
def evaluate(model, test_loader):
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            _, logits = model(input_ids)
            y_pred.extend(torch.argmax(logits, dim=1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))

# Main
if __name__ == "__main__":
    model = BERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-6)

    train(model, optimizer, train_loader, valid_loader)
    best_model = BERT().to(device)
    load_checkpoint(os.path.join(destination_folder, 'model.pt'), best_model)
    evaluate(best_model, test_loader)

