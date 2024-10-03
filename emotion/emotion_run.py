from transformers import ElectraForSequenceClassification, ElectraTokenizer
import json
import torch
import kss
from flask import Response, jsonify
from flask import request
model_path = "emotion/model"  # 학습된 모델 경로
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id_to_emotion = {0: '불안', 1: '놀람/당황', 2: '분노', 3: '슬픔', 4: '행복'}
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    predicted_emotion = id_to_emotion[predicted_class]
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    
    return predicted_emotion, confidence

def emotion_run():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Invalid input"}), 400
    
    text = request.json['text']
    print(text)
    # 문장 분리
    sentences = kss.split_sentences(text)
    
    results = []
    for sentence in sentences:
        emotion, confidence = predict_emotion(sentence)
        results.append({
            "sentence": sentence,
            "emotion": emotion,
            "confidence": confidence
        })
    
    # 감정별 문장 수 계산
    emotion_counts = {}
    for result in results:
        emotion = result['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    print(results, emotion_counts, )
    response_data = {
    "sentences": results,
    "emotion_counts": emotion_counts,
    "total_sentences": len(sentences)
}
    return jsonify(response_data)
  
