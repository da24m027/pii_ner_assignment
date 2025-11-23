import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import optuna
from optuna.trial import TrialState
import numpy as np
import logging

from dataset import PIIDataset, collate_batch
from labels import LABELS, LABEL2ID
from model import create_model

# Reduce verbosity
logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="microsoft/deberta-v3-small")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameter tuning arguments
    ap.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    ap.add_argument("--n_trials", type=int, default=10, help="Number of tuning trials")
    ap.add_argument("--tune_study_name", default="pii_tuning", help="Optuna study name")
    ap.add_argument("--tune_storage", default="sqlite:///optuna_study.db", help="Optuna storage")
    ap.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    return ap.parse_args()


def compute_metrics(predictions, labels, label_list):
    """Compute precision, recall, F1 for each entity type (optimized)"""
    true_predictions = []
    true_labels = []
    
    # Flatten and filter in one pass
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:  # Ignore padding
                true_predictions.append(pred)
                true_labels.append(label)
    
    # Quick F1 calculation (binary average for speed)
    if len(true_predictions) == 0:
        return 0.0
    
    # Count matches for non-O labels
    correct = sum(1 for p, l in zip(true_predictions, true_labels) if p == l and l != 0)
    predicted_entities = sum(1 for p in true_predictions if p != 0)
    true_entities = sum(1 for l in true_labels if l != 0)
    
    if predicted_entities == 0 or true_entities == 0:
        return 0.0
    
    precision = correct / predicted_entities
    recall = correct / true_entities
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def evaluate(model, dataloader, device, label_list):
    """Evaluate model on dev set"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = compute_metrics(all_predictions, all_labels, label_list)
    
    model.train()
    return avg_loss, macro_f1


def train_model(args, trial=None):
    """Train model with given hyperparameters"""
    
    # Hyperparameters (from trial or args)
    if trial is not None:
        lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_int("epochs", 3, 6)  # Reduced max epochs
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.05)
    else:
        lr = args.lr
        batch_size = args.batch_size
        epochs = args.epochs
        warmup_ratio = 0.1
        weight_decay = 0.01
    
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    dev_dl = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    model = create_model(args.model_name)
    model.to(args.device)
    model.train()
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    best_f1 = 0.0
    label_list = ["O"] + [f"B-{label}" for label in LABELS] + [f"I-{label}" for label in LABELS]
    
    # Disable tqdm during tuning
    use_tqdm = (trial is None) and (not args.quiet)
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        if use_tqdm:
            progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            progress_bar = train_dl
        
        for batch in progress_bar:
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if use_tqdm:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = running_loss / len(train_dl)
        
        # Evaluate on dev set
        dev_loss, dev_f1 = evaluate(model, dev_dl, args.device, label_list)
        
        if trial is None and not args.quiet:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
        
        # Save best model
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            if trial is None:  # Only save during normal training
                model.save_pretrained(args.out_dir)
                tokenizer.save_pretrained(args.out_dir)
                if not args.quiet:
                    print(f"  ✓ New best model saved! F1: {best_f1:.4f}")
        
        # Report to Optuna for pruning
        if trial is not None:
            trial.report(dev_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Clear CUDA cache to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_f1


def objective(trial, args):
    """Optuna objective function"""
    return train_model(args, trial=trial)


def run_hyperparameter_tuning(args):
    """Run Optuna hyperparameter tuning"""
    print(f"Starting hyperparameter tuning with {args.n_trials} trials...")
    
    # Create study
    study = optuna.create_study(
        study_name=args.tune_study_name,
        storage=args.tune_storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    
    # Optimize with timeout protection
    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            timeout=None,  # Remove timeout
            gc_after_trial=True,  # Garbage collect after each trial
            show_progress_bar=False  # Disable optuna progress bar
        )
    except KeyboardInterrupt:
        print("\nTuning interrupted by user")
    
    # Print results
    print("\n" + "="*50)
    print("Hyperparameter Tuning Complete!")
    print("="*50)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\nStatistics:")
    print(f"  Finished trials: {len(study.trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")
    
    if len(complete_trials) > 0:
        print(f"\nBest trial (F1: {study.best_trial.value:.4f}):")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Save best hyperparameters
        best_params_path = os.path.join(args.out_dir, "best_hyperparameters.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
        print(f"\nBest hyperparameters saved to {best_params_path}")
        
        # Train final model with best hyperparameters
        print("\n" + "="*50)
        print("Training final model with best hyperparameters...")
        print("="*50)
        
        args.lr = study.best_trial.params["lr"]
        args.batch_size = study.best_trial.params["batch_size"]
        args.epochs = study.best_trial.params["epochs"]
        
        final_f1 = train_model(args, trial=None)
        print(f"\nFinal model F1: {final_f1:.4f}")
        print(f"Model saved to {args.out_dir}")
    else:
        print("\nNo trials completed successfully.")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.tune:
        run_hyperparameter_tuning(args)
    else:
        print("Training with default/specified hyperparameters...")
        final_f1 = train_model(args, trial=None)
        print(f"\nTraining complete! Best F1: {final_f1:.4f}")
        print(f"Model saved to {args.out_dir}")


if __name__ == "__main__":
    main()