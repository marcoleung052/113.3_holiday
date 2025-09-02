## GPT2 護理記錄補全

這個專案使用 Hugging Face 的 GPT-2 模型，訓練於 MIMIC-III 的 Nursing Notes，並提供一個互動式補全介面，讓使用者逐字選擇生成內容，探索模型的語言能力與醫療語境掌握。
---
### 環境
- Ubuntu 22.04.1
- Python 3.9.13
- CUDA 12.4

---
### 環境需求
- PyTorch 2.6.0+cu124
- Transformers 4.56.0
- Datasets 4.0.0
- pandas 2.3.2

---
### code
[code](https://github.com/marcoleung052/113.3_holiday/blob/12ca47c0a4ffdc3cf420be7f89043b69b804c133/week4/gpt2_get_notes_final.py "游標顯示") 

---
### 步驟一：資料處理

從 `NOTEEVENTS.csv` 中篩選 `"CATEGORY" == "Nursing"` 的筆記，並清理文字：

```python
import pandas as pd

df = pd.read_csv("NOTEEVENTS.csv")
nursing_df = df[df["CATEGORY"] == "Nursing"]

def clean_text(text):
    return str(text).strip().replace("\n", " ").replace("\r", " ")

nursing_texts = nursing_df["TEXT"].dropna().apply(clean_text).tolist()

with open("nursing_corpus.txt", "w", encoding="utf-8") as f:
    for line in nursing_texts:
        f.write(line + "\n")
```

---

### 步驟二：模型訓練（GPT-2）

使用 Hugging Face 的 `Trainer` 訓練 GPT-2 模型：

```python
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_dict({"text": nursing_texts})

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-nursing",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
```

---

### 步驟三：互動式補全介面

使用者可輸入開頭句子，模型提供 5 個可能的下一個詞選項與機率，使用者逐步選擇直到完成句子：

```bash
python gpt2_nursing_interactive.py
```

範例互動：

```text
請輸入開頭句子：Pt required as additional 2 

請選擇下一個詞：
a. L (16.9%)
b. units (13.2%)
c. . (12.9%)
d. L (10.5%)
e. lit (4.1%)

請輸入選項或指令(re=重新生成, stop=結束)：b

目前句子：Pt required as additional 2 units
...
```

---

### 支援指令

| 指令 | 功能說明 |
|------|----------|
| `re` | 重新生成選項 |
| `stop` | 結束補全 |
| `a`~`e` | 選擇對應詞彙 |
