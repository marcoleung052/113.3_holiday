# 中文 GPT-2 文本補全實驗報告

本專案旨在比較多種中文 GPT-2 模型在文本補全任務中的表現，並分析不同 `max_length` 設定對補全結果的影響。

## 補全方法與比較

我們使用以下兩種補全方式進行測試：

### 1. CLI 模式

使用命令列介面輸入開頭句子，設定 `max_length` 參數進行補全。

```bash
python generate.py --text "他們" --max_length 50
```

### 模型比較
我們測試了以下幾個中文 GPT-2 模型：
| 模型名稱 | 補全風格 | 語意連貫性 | 創造力 | 備註 | 
| 原始 GPT-2 中文 | 偏向政治與歷史敘述 | 中等 | 低 | 語料偏向官方文本 | 
| ckiplab/gpt2-base-chinese | 敘述自然，偏向新聞風格 | 高 | 中 | 表現穩定 | 
| IDEA-CCNL/Wenzhong-GPT2-110M | 敘述跳躍，創意高 | 中 | 高 | 偶有語意混亂 | 
| uer/gpt2-chinese-cluecorpussmall | 偏向生活評論與食記 | 高 | 中 | 適合口語化應用 | 



📏 max_length 實測分析
我們針對 max_length 參數進行實驗，觀察補全結果的變化：
實驗設定
- 開頭句子：「他們」
- 測試 max_length：20、50、100
結果摘要
| max_length | 補全長度 | 語意完整性 | 重複率 | 備註 | 
| 20 | 短句，常中斷 | 低 | 低 | 適合快速預覽 | 
| 50 | 中等長度，語意較完整 | 中 | 中 | 建議預設值 | 
| 100 | 長句，語意完整但可能冗長 | 高 | 高 | 容易出現重複或語意漂移 | 


補全範例（max_length=50）
他們的菜味道和數據都很好，服務態度也都挺好，菜價也合理，所以生意很好，適合公司集體活動，公司請客。


📎 結論與建議
- 若需快速生成短句，建議使用 max_length=20。
- 若希望語意完整且自然，max_length=50 為較佳選擇。
- 模型選擇應依應用場景而定，生活化應用推薦 uer 模型，正式文本可選用 ckiplab 或原始 GPT-2。

📂 專案結構
├── generate.py        # CLI 補全腳本
├── app.py             # Flask Web 介面
├── models/            # 模型設定與載入
├── static/            # 前端資源
├── templates/         # HTML 模板
└── README.md          # 專案說明文件



🛠️ 環境需求
- Python 3.8+
- Transformers
- Flask
安裝方式：
pip install -r requirements.txt



📬 聯絡方式
如有任何問題或建議，歡迎開 issue 或聯絡作者。

如果你有 logo、範例圖片或想加入 badge（如 Hugging Face 模型連結），我也可以幫你加上！


