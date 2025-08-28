# 中文文本補全實驗報告

本專案旨在比較多種中文模型在文本補全任務中的表現，並分析不同模型訓練和prompt設定對補全結果的影響。

## 環境
- Ubuntu 22.04.1
- Python 3.9.13
- CUDA 12.4

## 環境需求
```txt
transformers==4.30.2
Flask==3.1.2
```

## Run code
這是我使用過的code：<br>
[ChatGLM3](https://github.com/marcoleung052/113.3_holiday/blob/6c24be4ab5bb1daf136aa98c94cebcb1f6301cd5/ChatGLM3.py "游標顯示") 這個是使用了ChatGLM3模型<br>
```txt
python ChatGLM3.py
```
[ChineseRoberta](https://github.com/marcoleung052/113.3_holiday/blob/6c24be4ab5bb1daf136aa98c94cebcb1f6301cd5/ChineseRoberta.py "游標顯示") 這個是使用了ChineseRoberta模型<br>
```txt
python ChineseRoberta.py
```

[RoFormer-Chinese](https://github.com/marcoleung052/113.3_holiday/blob/6c24be4ab5bb1daf136aa98c94cebcb1f6301cd5/RoFormer-Chinese.py "游標顯示") 這個是使用了RoFormer-Chinese模型<br>
```txt
python RoFormer-Chinese.py
```

[Taiwan‑LLM](https://github.com/marcoleung052/113.3_holiday/blob/6c24be4ab5bb1daf136aa98c94cebcb1f6301cd5/Taiwan%E2%80%91LLM.py "游標顯示") 這個是使用了Taiwan‑LLM模型<br>
```txt
python Taiwan‑LLM.py
```

[WoBERT](https://github.com/marcoleung052/113.3_holiday/blob/6c24be4ab5bb1daf136aa98c94cebcb1f6301cd5/WoBERT.py "游標顯示") 這個是使用了WoBERT模型<br>
```txt
python WoBERT.py
```

## 使用方式
### CLI 補全
快速命令列補全，適合批次處理或簡易測試。
```python
input_text = "他們"
inputids = tokenizer.encode(inputtext, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=20,
    numreturnsequences=3,
    do_sample=True,
    top_k=50,
    temperature=0.9,
    padtokenid=tokenizer.padtokenid
)

print(tokenizer.decode(output[0], skipspecialtokens=True))
```

## 模型比較
我們測試了以下幾個中文模型的效果：
| 模型              | 輸入長度影響             | 自然度 / 重複情況                   | 簡繁體支援                                  | 品質評估 |
|-------------------|--------------------------|--------------------------------------|---------------------------------------------|------------|
| ChatGLM3          | 效果佳                   | 高自然度，少重複                    | 簡體佳，繁體可用但較偏簡體語料風格         | ★★★★       |
| ChineseRoberta    | 短輸入可以，長輸入佳     | 自然度中等，短輸入易重複詞          | 簡體佳，繁體需轉換                          | ★★★        |
| RoFormerChinese   | 短輸入差，長輸入不佳     | 自然度低，易出現符號重複            | 簡體佳，繁體需轉換                          | ★          |
| Taiwan‑LLM        | 效果佳                   | 高自然度，少重複                    | 繁體佳，簡體可用                            | ★★★★★      |
| WoBERT            | 短輸入差，長輸入佳       | 自然度中等，適合詞語補全            | 簡體佳，繁體需轉換                          | ★★         |

## 模型分析
我們幾個中文模型效果的分析：
| 模型              | 任務形式             | 分析 |
|-------------------|--------------------------|------------|
| ChatGLM3          | Chat式生成 | 指令式 Chat 模型，對 prompt 依從性高，適合需控格式與語氣的應用 |
| ChineseRoberta    | Masked LM | 類 BERT 模型，對長上下文處理良好，短輸入語境不足時易詞語重複 |
| RoFormerChinese   | 自回歸生成（Rotary Position Embedding） | 解碼時易陷入符號 loop，可透過調整 decoding 參數改善 |
| Taiwan‑LLM        | Chat/生成式 | 對繁體中文有針對性優化，語料在地化程度高，對臺灣用語表現佳      |
| WoBERT            | Masked LM（BERT 架構） | Masked LM 架構，長文本詞彙預測精準，長句生成能力有限 |
