"""Script de diagnostic avancé pour identifier le problème exact"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from evaluate import load
import torch
import random

# Configuration
model_path = "artifacts/model_trainer/pegasus-samsum-model"
tokenizer_path = "artifacts/model_trainer/tokenizer"
data_path = "artifacts/data_transformation/samsum_dataset"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}\n")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Charger les données
dataset = load_from_disk(data_path)
test_data = dataset['test']

# Échantillonnage EXACTEMENT comme dans model_evaluation.py
random.seed(42)
test_size = len(test_data)
sample_size = 30
test_indices = random.sample(range(test_size), sample_size)
test_subset = test_data.select(test_indices)

print("="*80)
print(f"📊 TEST SUR {sample_size} EXEMPLES (EXACTEMENT COMME main.py)")
print("="*80)

# Extraire les colonnes EXACTEMENT comme dans le code fixé
dialogues = test_subset['dialogue']
summaries = test_subset['summary']

print(f"\n🔍 Type dialogues: {type(dialogues)}")
print(f"🔍 Type summaries: {type(summaries)}")
print(f"🔍 Longueur dialogues: {len(dialogues)}")
print(f"🔍 Longueur summaries: {len(summaries)}")

print(f"\n🔍 Premier dialogue (extrait): {dialogues[0][:100]}...")
print(f"🔍 Premier summary: {summaries[0]}")

# Simuler le traitement par batch
all_predictions = []
all_references = []
batch_size = 2

print(f"\n" + "="*80)
print("🔄 SIMULATION DU TRAITEMENT PAR BATCH")
print("="*80)

for i in range(0, len(dialogues), batch_size):
    batch_end = min(i + batch_size, len(dialogues))
    
    batch_texts = dialogues[i:batch_end]
    batch_summaries = summaries[i:batch_end]
    
    print(f"\n--- Batch {i//batch_size + 1} (indices {i} à {batch_end-1}) ---")
    print(f"🔍 Type batch_texts: {type(batch_texts)}")
    print(f"🔍 Longueur batch_texts: {len(batch_texts)}")
    
    if i == 0:  # Afficher détails pour le premier batch
        print(f"🔍 batch_texts: {batch_texts}")
        print(f"🔍 batch_summaries: {batch_summaries}")
    
    # Tokeniser
    inputs = tokenizer(batch_texts, max_length=1024, truncation=True, 
                      padding="max_length", return_tensors="pt")
    
    # Générer
    summary_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        length_penalty=0.8, num_beams=2, max_length=128
    )
    
    # Décoder
    decoded_summaries = [
        tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
        for s in summary_ids
    ]
    
    decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
    
    if i == 0:  # Afficher pour le premier batch
        print(f"🤖 decoded_summaries: {decoded_summaries}")
    
    all_predictions.extend(decoded_summaries)
    all_references.extend(batch_summaries)
    
    if i >= 4:  # Arrêter après 3 batches pour debug
        break

print(f"\n" + "="*80)
print("📊 CALCUL ROUGE")
print("="*80)

print(f"\n🔍 Total predictions: {len(all_predictions)}")
print(f"🔍 Total references: {len(all_references)}")
print(f"\n🔍 Type all_predictions: {type(all_predictions)}")
print(f"🔍 Type all_references: {type(all_references)}")

print(f"\n🔍 Première prediction: {all_predictions[0]}")
print(f"🔍 Première reference: {all_references[0]}")

# Calculer ROUGE
rouge_metric = load('rouge')
score = rouge_metric.compute(predictions=all_predictions, references=all_references)

print(f"\n" + "="*80)
print("📊 RÉSULTATS ROUGE (sur premiers 6 exemples)")
print("="*80)
for metric, value in score.items():
    print(f"{metric}: {value:.4f}")

print("\n" + "="*80)
print("💡 DIAGNOSTIC")
print("="*80)
if score['rouge1'] > 0.30:
    print("✅ Les scores sont bons ! Le problème est ailleurs dans le code.")
else:
    print("❌ Les scores sont mauvais. Il y a un problème avec les données ou le modèle.")
    print("\n🔍 Vérifiez si les prédictions et références sont cohérentes:")
    for i in range(min(3, len(all_predictions))):
        print(f"\nExemple {i+1}:")
        print(f"  Ref: {all_references[i][:100]}...")
        print(f"  Pred: {all_predictions[i][:100]}...")