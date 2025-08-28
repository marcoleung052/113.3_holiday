from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"  # 正確的模型 ID
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

prompt = "請用繁體中文把以下的句子補全10個字，給我3個選項後每一個選項都有對應的機率：他們"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.1
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))