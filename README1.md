# 中文 GPT-2 文本補全實驗報告

本專案旨在比較多種中文 GPT-2 模型在文本補全任務中的表現，並分析不同 `max_length` 設定對補全結果的影響。

## 📌 補全方法與比較

我們使用以下兩種補全方式進行測試：

### 1. CLI 模式

使用命令列介面輸入開頭句子，設定 `max_length` 參數進行補全。

```bash
python generate.py --text "他們" --max_length 50
