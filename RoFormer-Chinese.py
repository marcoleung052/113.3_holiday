from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "junnyu/roformer_chinese_char_base"  # 因果版本
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
他 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。
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
他 不
'''

prompt = "他們[MASK]不錯。"
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
他 不 。 ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ，
'''

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
今 天 下 班 我 打 算 ， ？? 。 ” “ “ 和 你 从 说 的 了 ！! ",. … 就 这 此 样 么 吧 像 : ： ） （ ( ) * # @ / \ ; & <
'''
