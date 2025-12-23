"""Script de debug pour identifier le problème ROUGE"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from evaluate import load
import torch

# Configuration
model_path = "artifacts/model_trainer/pegasus-samsum-model"
tokenizer_path = "artifacts/model_trainer/tokenizer"
data_path = "artifacts/data_transformation/samsum_dataset"

# Charger le modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Charger les données
dataset = load_from_disk(data_path)
test_data = dataset['test']

# Prendre juste 3 exemples pour le debug
print("\n" + "="*80)
print("🔍 TEST SUR 3 EXEMPLES")
print("="*80)

predictions = []
references = []

for i in range(3):
    dialogue = test_data[i]['dialogue']
    reference_summary = test_data[i]['summary']
    
    # Générer le résumé
    inputs = tokenizer([dialogue], max_length=1024, truncation=True, 
                      padding="max_length", return_tensors="pt")
    
    summary_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        length_penalty=0.8,
        num_beams=2,
        max_length=128
    )
    
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=True)
    
    print(f"\n--- Exemple {i+1} ---")
    print(f"📝 Dialogue (extrait): {dialogue[:100]}...")
    print(f"✅ Référence: {reference_summary}")
    print(f"🤖 Généré: {generated_summary}")
    
    # Vérifier les types
    print(f"\n🔍 Type référence: {type(reference_summary)}")
    print(f"🔍 Type généré: {type(generated_summary)}")
    
    predictions.append(generated_summary)
    references.append(reference_summary)

# Tester ROUGE
print("\n" + "="*80)
print("📊 TEST MÉTRIQUE ROUGE")
print("="*80)

rouge_metric = load('rouge')

print(f"\n🔍 Predictions: {predictions}")
print(f"🔍 References: {references}")

# Calculer ROUGE
score = rouge_metric.compute(predictions=predictions, references=references)

print("\n" + "="*80)
print("📊 RÉSULTATS ROUGE")
print("="*80)
for metric, value in score.items():
    print(f"{metric}: {value:.4f}")

print("\n✅ Si les scores sont > 0.30, le problème est dans model_evaluation.py")
print("❌ Si les scores sont < 0.05, le problème est dans le modèle ou les données")