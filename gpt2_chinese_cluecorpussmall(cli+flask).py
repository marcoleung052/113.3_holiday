from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, BertTokenizerFast

model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
tokenizer = BertTokenizerFast.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

input_text = "他們"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=3,
    do_sample=True,
    top_k=50,
    temperature=0.9,
    pad_token_id=tokenizer.pad_token_id
)

print(tokenizer.decode(output[0], skip_special_tokens=True))

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    input_text = data.get("text", "")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 50,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"result": result})

def generate_options_with_scores(prompt, num_options=5):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 50,
        num_return_sequences=num_options,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True
    )

    results = []
    for sequence, score_set in zip(outputs.sequences, outputs.scores):
        # 計算整體 log-probability，越大代表越高機率
        logprobs = [score.max().item() for score in score_set]
        total_score = sum(logprobs)

        # 解碼文字
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        results.append((text, total_score))

    return results

# 主程式迴圈
def generate_options(prompt, num_options=5):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 50,
        num_return_sequences=num_options,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return results

while True:
  text = input("請輸入文字（按 Enter 結束）：")
  if not text.strip():
    break
  completions = generate_options_with_scores(text)
  print("GPT-2 補全建議：")
  for i, c in enumerate(completions):
    print(f"{i+1}. {c}")
  print("-" * 40)

def generate_options_with_scores(prompt, num_options=5):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 10,
        num_return_sequences=num_options,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True
    )

    results = []
    for sequence, score_set in zip(outputs.sequences, outputs.scores):
        # 解碼並移除原始 prompt 部分
        prompt_len = len(input_ids[0])
        full_text = tokenizer.decode(sequence, skip_special_tokens=True)
        continuation = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()

        # 簡單 log-prob 統計（可自定）
        logprobs = [score.max().item() for score in score_set]
        total_score = sum(logprobs)

        results.append((continuation.replace(" ", ""), total_score))

    return results

current_text = ""

while True:
    if not current_text:
        user_input = input("請輸入開頭句子（或 Enter 離開）：").strip()
        if not user_input:
            break
        current_text = user_input

    completions = generate_options_with_scores(current_text)
    print("\n補全建議:")
    for i, (cont, score) in enumerate(completions):
        print(f"{i + 1}. {cont}  ({score:.2f})")

    action = input("\n輸入編號選擇句子，或輸入 next / again： ").strip().lower()

    if action == "next":
        current_text = ""
        continue
    elif action == "again":
        continue
    elif action.isdigit():
        idx = int(action) - 1
        if 0 <= idx < len(completions):
            selected = completions[idx][0]
            current_text += selected
            print(f"\n已接續句子：{current_text}\n")
        else:
            print("選項超出範圍")
    else:
        print("請輸入合法的選項")

    print("-" * 50)