import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from load_dict import load_cmu_dict, split_data
from CMUDataset import CMUDataset
from evaluation import evaluate_per

# Загружаем данные
words, phonemes = load_cmu_dict('cmudict-07b.txt')
train_words, test_words, train_phonemes, test_phonemes = split_data(words, phonemes)

# Инициализация модели и токенайзера
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Подготовка датасетов и загрузчиков
train_dataset = CMUDataset(train_words, train_phonemes, tokenizer)
test_dataset = CMUDataset(test_words, test_phonemes, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Подключение к устройству
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Оптимизатор
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Обучение модели
epochs = 5
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    test_loss = 0

    progress_bar_test = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch in progress_bar_test:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Epoch {epoch + 1}, Test Loss: {avg_test_loss:.4f}")

# Сохранение модели
model_save_path = 't5_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Модель сохранена по пути: {model_save_path}")

# Оценка модели на тестовой выборке
per = evaluate_per(model, tokenizer, test_loader)
print(f"Phoneme Error Rate (PER) on test set: {per:.2f}")
