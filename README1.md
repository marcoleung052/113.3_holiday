# 中文 GPT-2 文本補全實驗報告

本專案旨在比較多種中文 GPT-2 模型在文本補全任務中的表現，並分析不同 `max_length` 設定對補全結果的影響。

## 使用方式總覽
### 1. CLI 補全
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

### 2. Gradio Web 介面
提供即時補全建議，適合展示與互動。
```python
def autocomplete(text):
    inputids = tokenizer.encode(text, returntensors="pt")
    output = model.generate(
        input_ids,
        max_length=20,
        numreturnsequences=1,
        norepeatngram_size=2,
        top_k=50,
        temperature=1.0,
        do_sample=True,
        padtokenid=tokenizer.eostokenid
    )
    return [tokenizer.decode(o, skipspecialtokens=True) for o in output]

gr.Interface(
    fn=autocomplete,
    inputs="text",
    outputs=gr.Textbox(type="text", label="Completions", lines=3),
    live=True
).launch()
```
### 3. Flask API 補全

提供 RESTful API，可整合至前端或其他系統。

```python
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    input_text = data.get("text", "")
    inputids = tokenizer.encode(inputtext, return_tensors="pt")
    output = model.generate(
        input_ids,
        maxlength=len(inputids[0]) + 50,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        padtokenid=tokenizer.padtokenid
    )
    result = tokenizer.decode(output[0], skipspecialtokens=True)
    return jsonify({"result": result})
```

### 4. Flask 選句互動補全（含 log-probability 評分）

提供多個補全選項並評分，使用者可逐句選擇接續內容。

```python
def generateoptionswithscores(prompt, numoptions=5):
    inputids = tokenizer.encode(prompt, returntensors="pt")
    outputs = model.generate(
        input_ids,
        maxlength=len(inputids[0]) + 50,
        numreturnsequences=num_options,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        padtokenid=tokenizer.padtokenid,
        output_scores=True,
        returndictin_generate=True
    )

    results = []
    for sequence, score_set in zip(outputs.sequences, outputs.scores):
        logprobs = [score.max().item() for score in score_set]
        total_score = sum(logprobs)
        text = tokenizer.decode(sequence, skipspecialtokens=True)
        results.append((text, total_score))

    return results
```

互動流程：

```python
while True:
    text = input("請輸入文字（按 Enter 結束）：")
    if not text.strip():
        break
    completions = generateoptionswith_scores(text)
    print("GPT-2 補全建議：")
    for i, c in enumerate(completions):
        print(f"{i+1}. {c}")
    print("-" * 40)
```





## 補全方法與比較
我們針對補全方法進行實驗，觀察補全結果的變化：
| 技術方式 | 優點 | 缺點 |
| ---- | ---- | ---- |
| CLI 單次補全 (命令列) | - 寫法簡單、適合快速測試模型<br>- 支援多個句子生成<br>- 容易嵌入到其他流程中 | - 無互動性，只能一次性生成<br>- 無候選選項或信心分數<br>- 無法持續接續或擴展語境 |
| Gradio 即時網頁補全 | - 使用者介面直觀、易上手<br>- 即時補全、自動更新<br>- 可本地或遠端部署 | - 僅產生單句，互動性低<br>- 無選項選擇、無接句功能<br>- 輸出長度與控制不夠彈性 |
| Flask API + CLI 補全迴圈 | - 有互動輸入、選擇與續句能力<br>- 顯示 log-prob 信心分數<br>- 可拓展成 Web API 或 GUI 工具 | - 使用門檻稍高（需理解 CLI 流程）<br>- 沒有圖形界面（需手動建 GUI）<br>- 結果無法直接儲存或複製分享 |

## max_length 實測分析
我們針對 max_length 參數進行實驗，觀察補全結果的變化：
實驗設定
- 開頭句子：「他們」
- 測試 max_length：20、50
  
結果摘要
| 長度設定 | 輸出風格 | 流暢度 | 結尾中斷感 |
| :----: | :----- | :----- | :----- |
| max_length = 20 | 通常僅一兩句，偏口語 | 中等（需良好接句） | 結尾通常突然結束或缺詞 |
| max_length = 50 | 可構成段落，有情境 | 高（語意連貫） | 結尾略突兀 |

補全範例（max_length=50）
他們的菜味道和數據都很好，服務態度也都挺好，菜價也合理，所以生意很好，適合公司集體活動，公司請客。

## 模型比較
我們測試了以下幾個中文 GPT-2 模型：
## 🤖 模型比較

| 模型名稱 | 風格傾向 | 簡繁體支援 | 品質評估 | 備註 |
|---|---|---|---|---|
| 原始 GPT-2 中文 | 混亂、口語不通 | 不能穩定處理繁體，易出錯 | ★ | 偏實驗用途，非實際場景推薦 |
| ckiplab/gpt2-base-chinese | 清晰分詞、風格制式 | 強繁體支援，分詞較穩定 | ★★ | 適合繁體任務，輸出比較偏向政治 |
| IDEA-CCNL/Wenzhong-GPT2-110M | 敘事感強，有創意但語法不穩定 | 偏簡體，繁體輸入會轉寫或失準 | ★★★ | 適合模擬口語或情境式生成|
| uer/gpt2-chinese-cluecorpussmall | 口語自然、近網民語言 | 偏繁體，但輸出有時候會出現簡體 | ★★★ | 美食評論風格明顯，實用性佳 |

## 環境需求
