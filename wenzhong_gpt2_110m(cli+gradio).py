from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'IDEA-CCNL/Wenzhong-GPT2-110M'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

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

import gradio as gr

def autocomplete(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=20,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        temperature=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in output]

gr.Interface(fn=autocomplete, inputs="text", outputs=gr.Textbox(type="text", label="Completions", lines=3), live=True).launch()