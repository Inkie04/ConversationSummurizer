from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Charger le modèle
model_path = "artifacts/model_trainer/pegasus-samsum-model"
tokenizer_path = "artifacts/model_trainer/tokenizer"

print("📥 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ Modèle chargé sur {device}!\n")

# Test 1: Conversation simple
dialogue1 = """
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
"""

# Test 2: Conversation professionnelle
dialogue2 = """
John: Hey Sarah, did you finish the project report?
Sarah: Yes! I just sent it to you. Did you review the budget section?
John: I did. The numbers look good. Should we schedule a meeting with the team?
Sarah: Great idea. How about tomorrow at 2 PM?
John: Perfect. I'll send the calendar invite.
"""

def test_summary(dialogue, title):
    print("=" * 70)
    print(f"🧪 TEST: {title}")
    print("=" * 70)
    print("DIALOGUE:")
    print(dialogue)
    print("\n🤖 GÉNÉRATION DU RÉSUMÉ...\n")
    
    inputs = tokenizer(dialogue, max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=8,
        length_penalty=0.8
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("📝 RÉSUMÉ GÉNÉRÉ:")
    print(summary)
    print("=" * 70)
    print()

# Tester
test_summary(dialogue1, "Conversation simple - Cookies")
test_summary(dialogue2, "Conversation professionnelle - Réunion")

print("\n💡 Si les résumés ont du sens, votre modèle fonctionne bien!")
print("💡 Les scores ROUGE bas sont probablement dus au petit nombre d'exemples d'évaluation.")