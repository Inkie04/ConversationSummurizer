from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from evaluate import load

import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig
import random
import gc




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                               batch_size=2, device="cuda" if torch.cuda.is_available() else "cpu", 
                               column_text="article", 
                               column_summary="highlights"):
        
        all_predictions = []
        all_references = []
        
        # Extraire les colonnes
        dialogues = dataset[column_text]
        summaries = dataset[column_summary]
        num_samples = len(dialogues)
        
        # Traiter par batches
        for i in tqdm(range(0, num_samples, batch_size)):
            # Extraire le batch actuel
            batch_end = min(i + batch_size, num_samples)
            
            batch_texts = dialogues[i:batch_end]
            batch_summaries = summaries[i:batch_end]
            
            # Tokeniser
            inputs = tokenizer(
                batch_texts, 
                max_length=1024, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            # Générer les résumés
            summary_ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device), 
                length_penalty=0.8, 
                num_beams=2,
                max_length=128
            )
            
            # Décoder les résumés (sans la ligne problématique)
            decoded_summaries = [
                tokenizer.decode(
                    s, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                for s in summary_ids
            ]
            
            # ✅ LIGNE PROBLÉMATIQUE RETIRÉE
            # Cette ligne causait l'insertion d'espaces entre chaque caractère
            # decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            # Libérer la mémoire
            del summary_ids
            torch.cuda.empty_cache()
            gc.collect()
            
            # Accumuler les résultats
            all_predictions.extend(decoded_summaries)
            all_references.extend(batch_summaries)
        
        # Calculer ROUGE
        score = metric.compute(predictions=all_predictions, references=all_references)
        
        return score


    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # Charger les données
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
  
        rouge_metric = load('rouge')

        # Échantillonnage de 30 exemples
        random.seed(42)
        test_size = len(dataset_samsum_pt['test'])
        sample_size = 30
        
        test_indices = random.sample(range(test_size), sample_size)
        test_subset = dataset_samsum_pt['test'].select(test_indices)

        print(f"\n📊 Évaluation ROUGE sur {sample_size} exemples aléatoires (total dataset: {test_size})")
        print("⏳ Temps estimé: ~5-8 minutes...\n")

        score = self.calculate_metric_on_test_ds(
            test_subset,
            rouge_metric, 
            model_pegasus, 
            tokenizer, 
            batch_size=2,
            device=device,
            column_text='dialogue', 
            column_summary='summary'
        )

        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame([rouge_dict], index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)
        
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE L'ÉVALUATION ROUGE")
        print(f"    (Échantillon aléatoire de {sample_size}/{test_size} exemples)")
        print("="*60)
        for metric, value in rouge_dict.items():
            print(f"{metric}: {value:.4f}")
        print("="*60)
        print(f"\n✅ Métriques sauvegardées dans: {self.config.metric_file_name}")
        print(f"💡 Scores attendus: rouge1 ~0.40-0.47")
        
        return rouge_dict