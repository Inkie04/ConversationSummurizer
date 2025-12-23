from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch
import random

# Charger
model_path = "artifacts/model_trainer/pegasus-samsum-model"
tokenizer_path = "artifacts/model_trainer/tokenizer"
data_path = "artifacts/data_transformation/samsum_dataset"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

dataset = load_from_disk(data_path)

# Prendre 5 exemples du test set
random.seed(42)
indices = random.sample(range(len(dataset['test'])), 5)

print("="*80)
print("🔍 COMPARAISON: RÉSUMÉS GÉNÉRÉS vs RÉSUMÉS DE RÉFÉRENCE")
print("="*80)

for idx in indices:
    example = dataset['test'][idx]
    
    dialogue = example['dialogue']
    reference_summary = example['summary']
    
    # Générer résumé
    inputs = tokenizer(dialogue, max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=8,
        length_penalty=0.8
    )
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print(f"\n{'='*80}")
    print(f"EXEMPLE {idx + 1}")
    print(f"{'='*80}")
    print(f"\n📝 DIALOGUE:")
    print(dialogue[:300] + "..." if len(dialogue) > 300 else dialogue)
    print(f"\n✅ RÉSUMÉ DE RÉFÉRENCE (dataset):")
    print(reference_summary)
    print(f"\n🤖 RÉSUMÉ GÉNÉRÉ (votre modèle):")
    print(generated_summary)
    print(f"\n{'='*80}")

print("\n💡 Analysez: Les résumés générés sont-ils cohérents mais formulés différemment?")