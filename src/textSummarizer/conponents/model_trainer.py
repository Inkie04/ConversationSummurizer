from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        # Vérifier si le modèle existe déjà
        model_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_path = os.path.join(self.config.root_dir, "tokenizer")
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            print(f"✅ Model already trained and exists at {model_path}")
            print("⏭️  Skipping training stage...")
            return
        
        # Si le modèle n'existe pas, continuer avec l'entraînement
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=1, 
            warmup_steps=500,
            per_device_train_batch_size=1, 
            per_device_eval_batch_size=1,
            weight_decay=0.01, 
            logging_steps=10,
            eval_strategy='steps', 
            eval_steps=500, 
            save_steps=1e6,
            gradient_accumulation_steps=16
        ) 

        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"], 
            eval_dataset=dataset_samsum_pt["validation"]
        )
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))