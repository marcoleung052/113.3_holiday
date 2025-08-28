from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "hfl/chinese-roberta-wwm-ext"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "他們"
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)
print(tok.decode(outputs[0], skip_special_tokens=True))

'''
他 們 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。
'''

prompt = "他們不錯"
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.5
)
print(tok.decode(outputs[0], skip_special_tokens=True))

'''
他 們 不 錯 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的
'''

prompt = "他們[MASK]不錯"
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.5
)
print(tok.decode(outputs[0], skip_special_tokens=True))

'''
他 們 不 錯 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的'''

prompt = "他們不錯"
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.5
)
print(tok.decode(outputs[0], skip_special_tokens=True))

'''
他 們 不 錯 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的'''

prompt = "今天下班後我打算"
inputs = tok(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=2.5
)
print(tok.decode(outputs[0], skip_special_tokens=True))

'''
今 天 下 班 後 我 打 算 的 出 上 一 次 回 家 去 。 ， 然 再 后 是 就 到 了 了 这 里 儿 吧 呢 ？ ！ ~'''