from transformers import pipeline

# Load the emotion classification model
emotion_model = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # Get all emotion predictions
)

# Get user input
text = input("Type something to detect the emotion: ")

# Get predictions and fix the nested list
results = emotion_model(text)[0]  # <--- this is the fix

# Sort by highest score
sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

# Print top emotion
top_emotion = sorted_results[0]
print(f"\nDetected Emotion: {top_emotion['label']} (Confidence: {top_emotion['score']:.2f})")

# Print all emotions
print("\nFull Emotion Scores:")
for res in sorted_results:
    print(f"{res['label']}: {res['score']:.2f}")
