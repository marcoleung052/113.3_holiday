from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

model_id = "junnyu/wobert_chinese_plus_base"
tok = AutoTokenizer.from_pretrained(model_id)
mlm = AutoModelForMaskedLM.from_pretrained(model_id)

nlp = pipeline("fill-mask", model=mlm, tokenizer=tok)

text = "他們[MASK]。"
for pred in nlp(text, top_k=5):
    print(pred["sequence"], pred["score"])

'''
他 。 0.1483561247587204
他 然 。 0.03041015937924385
他 了 。 0.029545851051807404
他 怒 。 0.015772942453622818
他 他 。 0.014552016742527485
'''

text = "他們不錯[MASK]。"
for pred in nlp(text, top_k=5):
    print(pred["sequence"], pred["score"])

'''
他 不 他 。 0.055230073630809784
他 不 我 。 0.05386219546198845
他 不 。 0.053843554109334946
他 不 意 。 0.01851041615009308
他 不 啊 。 0.016992108896374702
'''

text = "他們[MASK]不錯。"
for pred in nlp(text, top_k=5):
    print(pred["sequence"], pred["score"])

'''
他 而 不 。 0.4198865294456482
他 不 。 0.18869413435459137
他 然 不 。 0.05887201428413391
他 也 不 。 0.038030028343200684
他 言 不 。 0.013068684376776218
'''

text = "這家店的牛肉麵[MASK]。"
for pred in nlp(text, top_k=5):
    print(pred["sequence"], pred["score"])

'''
家 店 的 牛肉 不错 。 0.24917654693126678
家 店 的 牛肉 子 。 0.07025124132633209
家 店 的 牛肉 好吃 。 0.06087698042392731
家 店 的 牛肉 。 0.033309418708086014
家 店 的 牛肉 家 。 0.0241136085242033
'''
