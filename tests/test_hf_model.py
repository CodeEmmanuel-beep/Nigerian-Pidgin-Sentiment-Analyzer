from transformers import pipeline

classifier = pipeline("text-classification", model="withus/afro-xlmr-weighted")

result = classifier("comot for here")
print(result)
