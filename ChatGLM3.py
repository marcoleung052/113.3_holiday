from transformers import AutoTokenizer, AutoModel

model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto").eval()

history = []
prompt = "請用繁體中文補全10個字，給我3個選項以後每一個選項都有對應的機率：他們"
response, history = model.chat(tokenizer, prompt, history=history)
print(response)

'''
以下是三個選項：

A. 他們是
B. 他們會
C. 他們在

機率分別為：

A. 40%
B. 30%
C. 30%
'''