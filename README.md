# Phonemes-generation
Этот проект обучает и тестирует модель на основе архитектуры T5 для генерации фонем для слов из словаря CMU Pronouncing Dictionary (cmudict). CMU Dict содержит английские слова и их фонемные транскрипции, записанные в ARPABET.

В процессе проекта:

* Мы загружаем данные из cmudict-07b.txt.
* Подготавливаем данные для обучения модели.
* Обучаем модель T5 для генерации фонем из входного слова.
* Тестируем модель и визуализируем метрику лосса (потерь) для процесса обучения и валидации.

## Модель T5

Модель T5, обученная для генерации фонем, доступна для скачивания по следующей ссылке:

[Скачать модель T5 (230MB)](https://drive.google.com/file/d/1_Q7dAVE5pJ0D2ZXpOKychX7j0K2NUDeN/view?usp=sharing)

## Структура кода
### Импорт библиотек
```
import torch
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
```
Этот блок импортирует необходимые библиотеки:

* transformers — для использования моделей и токенизаторов T5.
* tqdm — для отображения прогресса в процессе обучения.
* matplotlib — для визуализации результатов.
* 

### Функция загрузки данных
```
def load_cmu_dict(file_path):
    words = []
    phonemes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            parts = line.strip().split()
            if len(parts) > 1:
                word = parts[0]
                phoneme_seq = parts[1:]
                words.append(word)
                phonemes.append(phoneme_seq)
    return words, phonemes
```
Эта функция загружает данные из файла cmudict-07b.txt. Она читает файл построчно, извлекает слова и соответствующие им фонемы, а затем возвращает два списка: один для слов, другой — для фонем.

### Разделение данных на тренировочные и тестовые

```
def split_data(words, phonemes):
    return train_test_split(words, phonemes, train_size=0.8, test_size=0.2, random_state=42)

```


### Определение класса датасета

```
class CMUDataset(Dataset):
    def __init__(self, words, phonemes, tokenizer, max_length=128):
        self.words = words
        self.phonemes = phonemes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        phoneme = " ".join(self.phonemes[idx])

        input_ids = self.tokenizer(word, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).input_ids
        labels = self.tokenizer(phoneme, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).input_ids

        return {'input_ids': input_ids.squeeze(), 'labels': labels.squeeze()}
```

Этот класс наследуется от PyTorch Dataset и используется для подготовки данных к обучению модели. Он берет слово и его транскрипцию и преобразует их в подходящий для модели формат с помощью токенизатора.


### Инициализация модели и токенизатора

```
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```
Здесь загружаются предобученные модель и токенизатор T5-small из библиотеки transformers.


### Создание загрузчиков данных (DataLoader)

```
train_dataset = CMUDataset(train_words, train_phonemes, tokenizer)
test_dataset = CMUDataset(test_words, test_phonemes, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)
```
Здесь мы создаем объекты DataLoader для обучающей и тестовой выборок с заданным размером батча 8.


### Обучение модели
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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
```
Цикл обучения модели проходит через несколько эпох. В каждой эпохе модель обучается на тренировочных данных, а затем проверяется на тестовой выборке. Результаты потерь сохраняются для дальнейшего анализа.

Как видно на изображении, лосс у нас падает, что может сказать о том, что обучение модели проходит хорошо
![image](https://github.com/user-attachments/assets/51b760f9-31f4-4606-8a09-a8030625eb82)


### Оценка модели: Phoneme Error Rate (PER)

Phoneme Error Rate (PER) — это метрика, которая измеряет процент фонем, которые были неправильно предсказаны моделью по сравнению с истинными значениями. PER рассчитывается как сумма ошибок (замен, вставок и удалений) деленная на общее количество фонем в истинной последовательности.

В коде предоставлена функция error_rate, которая вычисляет PER для двух последовательностей фонем: истинной (timit) и предсказанной (asr).


Аргументы:

* timit: истинная последовательность фонем (строка или список).
* asr: предсказанная последовательность фонем (строка или список).
* phn: логическое значение, указывающее, следует ли разделять последовательности по пробелам или другим разделителям.
Процесс:

1. Инициализация матриц: Создаются матрицы для хранения стоимости операций редактирования и для отслеживания изменений.
2. Заполнение матриц: Заполняются значения стоимости для различных операций (замена, вставка, удаление).
3. Обратный проход: По матрице отслеживания осуществляется обратный проход, который подсчитывает количество ошибок.
4. Расчет PER: Итоговое значение PER рассчитывается на основе количества ошибок и длины истинной последовательности.

### Оценка модели
Для оценки модели используется функция evaluate_per, которая принимает модель, токенизатор и загрузчик тестовой выборки. Она рассчитывает средний PER по всем последовательностям в тестовом наборе данных.

В нашем случае PER модели, обученной на cmudict составила 0.67, что очень слабый показатель, но тем не менее при эмпирической оценке сгенерированные фонемы похожи на действительные
![image](https://github.com/user-attachments/assets/c91eb922-8c53-49d2-9f6a-fbc668a64baa)
