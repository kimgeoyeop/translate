from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델/토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

# 사용자 입력
src_text = input("번역할 한국어 문장 입력: ")

# 토크나이즈
inputs = tokenizer(src_text, return_tensors="pt")

# 번역 실행
translated = model.generate(**inputs, max_length=40)

# 결과 디코딩
result = tokenizer.decode(translated[0], skip_special_tokens=True)

print("\n📘 번역 결과:")
print(result)
