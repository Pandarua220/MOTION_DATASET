"""
train_kfold_cv.py
========================================================
Batch Training Script with 9-Fold Cross-Validation
Strategy: 7 Folds Train, 1 Fold Validation, 1 Fold Test
"""

import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from itertools import product
from typing import Dict
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.model_selection import KFold
import gc

from dataloader import SleepSequenceDataset
from model import create_model
from utils import export_results_unified

def get_all_groups(config):
    """
    Load dataset once to discover all Subject/Group IDs.
    Returns a sorted list of unique group IDs.
    """
    print("Scanning dataset for groups...")
    # Initialize with minimal overhead just to scan IDs
    temp_ds = SleepSequenceDataset(
        subjects=None, # Load all
        channels=list(range(config["num_channels"])),
        window_radius=config["window_radius"],
        sequence_length=1, 
        task="multi",
        normalize=None, 
        pca_components=None,
        augment=False,
        is_training=False
    )
    
    unique_groups = np.unique(temp_ds.subject_ids)
    print(f"Found {len(unique_groups)} unique subject groups: {unique_groups}")
    
    del temp_ds
    gc.collect()
    
    return unique_groups

def run_kfold_cv(config: Dict, task: str, device: torch.device, all_groups: np.ndarray, n_splits: int = 9) -> Dict:
    """
    Execute 9-Fold CV with Train/Val/Test Split (7/1/1).
    """
    if task == "multi":
        num_classes = 3
    else: 
        num_classes = 2
        
    # Initialize KFold splitter
    # We use random_state to ensure the 'next fold' logic is consistent
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Generate all split indices first so we can rotate them
    # splits is a list of (train_indices, test_indices)
    # Note: In standard KFold, 'test_indices' is the chunk held out.
    splits = list(kf.split(all_groups))
    
    # Storage
    cv_results = defaultdict(list)
    all_confusion_matrices = []
    all_predictions = []
    all_true_labels = []
    
    # Common Dataset Args
    common_args = {
        "channels": list(range(config["num_channels"])),
        "window_radius": config["window_radius"],
        "sequence_length": config["sequence_length"] if config["model"] != 'mlp' else 1,
        "task": task,
        "roi_size": config["roi_size"],
        "normalize": "standard",
        "pca_components": config["pca_components"] if config["pca_enabled"] else 0,
    }

    print(f"\nStarting {n_splits}-Fold CV (7 Train / 1 Val / 1 Test)...")

    # Iterate Folds
    for fold_idx in range(n_splits):
        # --- 1. Determine Groups for this Fold ---
        # Current 'test' part of KFold is our TEST set
        _, test_indices = splits[fold_idx]
        
        # The 'test' part of the NEXT fold is our VALIDATION set (Cyclic)
        val_fold_idx = (fold_idx + 1) % n_splits
        _, val_indices = splits[val_fold_idx]
        
        # TRAIN is everything else
        # Create masks to select the remaining groups
        total_mask = np.ones(len(all_groups), dtype=bool)
        total_mask[test_indices] = False
        total_mask[val_indices] = False
        
        train_groups = all_groups[total_mask]
        val_groups = all_groups[val_indices]
        test_groups = all_groups[test_indices]
        
        print(f"\n[Task: {task}] Fold {fold_idx+1}/{n_splits}")
        print(f"  Train Groups: {len(train_groups)}, Val Groups: {len(val_groups)}, Test Groups: {len(test_groups)}")
        print(f"  Test Group IDs: {test_groups}")
        
        # --- 2. Create Datasets ---
        
        # A. Train Dataset (Fits Scaler & PCA)
        train_dataset = SleepSequenceDataset(
            subjects=list(train_groups),
            is_training=True,
            augment=True, # Augmentation only on Train
            **common_args
        )
        
        # Get fitted transformers from Training set
        scaler = train_dataset.get_scaler()
        pca_model = train_dataset.get_pca_model()
        
        # B. Validation Dataset (Uses Train Scaler/PCA)
        val_dataset = SleepSequenceDataset(
            subjects=list(val_groups),
            is_training=False,
            augment=False,
            scaler=scaler,
            pca_model=pca_model,
            **common_args
        )
        
        # C. Test Dataset (Uses Train Scaler/PCA)
        test_dataset = SleepSequenceDataset(
            subjects=list(test_groups),
            is_training=False,
            augment=False,
            scaler=scaler,
            pca_model=pca_model,
            **common_args
        )
        
        # --- 3. DataLoaders ---
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, 
            num_workers=8, pin_memory=True, drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False, 
            num_workers=8, pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False, 
            num_workers=8, pin_memory=True
        )
        
        # Check input size
        try:
            sample_batch = next(iter(train_loader))
            input_size = sample_batch[0].shape[-1]
        except StopIteration:
            print("  Warning: Empty training batch. Skipping fold.")
            continue

        # --- 4. Model Setup ---
        model = create_model(
            model_type=config["model"],
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            use_mask=True
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"], 
            weight_decay=config["weight_decay"]
        )
        
        # Early Stopping Variables
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None
        
        # --- 5. Training Loop ---
        for epoch in range(config["num_epochs"]):
            # Train
            model.train()
            train_losses = []
            for sequences, labels, masks in train_loader:
                sequences, labels, masks = sequences.to(device), labels.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(sequences, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            if epoch > 25:
                # Validation (For Early Stopping)
                model.eval()
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for sequences, labels, masks in val_loader:
                        sequences, labels, masks = sequences.to(device), labels.to(device), masks.to(device)
                        outputs = model(sequences, masks)
                        preds = outputs.argmax(dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                val_acc = accuracy_score(val_labels, val_preds)
                
                # Check Early Stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                    # Optional: print improvement
                    # print(f"    Epoch {epoch}: New Best Val Acc: {val_acc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= config["patience"]:
                        print(f"    Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
                        break
        
        # --- 6. Final Test Evaluation ---
        # Load best model based on Validation performance
        if best_state:
            model.load_state_dict(best_state)
            
        model.eval()
        fold_test_preds = []
        fold_test_labels = []
        
        with torch.no_grad():
            for sequences, labels, masks in test_loader:
                sequences, labels, masks = sequences.to(device), labels.to(device), masks.to(device)
                outputs = model(sequences, masks)
                preds = outputs.argmax(dim=1)
                fold_test_preds.extend(preds.cpu().numpy())
                fold_test_labels.extend(labels.cpu().numpy())
        
        # Store Fold Results
        fold_test_preds = np.array(fold_test_preds)
        fold_test_labels = np.array(fold_test_labels)
        
        if len(fold_test_labels) > 0:
            acc = accuracy_score(fold_test_labels, fold_test_preds)
            f1 = f1_score(fold_test_labels, fold_test_preds, average='weighted')
            kappa = cohen_kappa_score(fold_test_labels, fold_test_preds)
            mcc = matthews_corrcoef(fold_test_labels, fold_test_preds)
            
            print(f"  Fold Result -> Acc: {acc:.4f}, Kappa: {kappa:.4f}")
            
            cv_results['accuracy'].append(acc)
            cv_results['f1'].append(f1)
            cv_results['kappa'].append(kappa)
            cv_results['mcc'].append(mcc)
            
            all_confusion_matrices.append(confusion_matrix(fold_test_labels, fold_test_preds, labels=range(num_classes)))
            all_predictions.extend(fold_test_preds)
            all_true_labels.extend(fold_test_labels)
        
        # Cleanup
        del train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # --- 7. Aggregate Results ---
    total_cm = np.sum(all_confusion_matrices, axis=0) if all_confusion_matrices else np.zeros((num_classes, num_classes))
    
    if task == "multi":
        class_names = ["NREM", "REM", "Wake"]
    elif task == "binary":
        class_names = ["Sleep", "Wake"]
    else:
        class_names = ["NREM", "REM+Wake"]

    results = {
        "accuracy": (np.mean(cv_results['accuracy']), np.std(cv_results['accuracy'])),
        "f1": (np.mean(cv_results['f1']), np.std(cv_results['f1'])),
        "kappa": (np.mean(cv_results['kappa']), np.std(cv_results['kappa'])),
        "mcc": (np.mean(cv_results['mcc']), np.std(cv_results['mcc'])),
        "conf_mat": total_cm,
        "predictions": np.array(all_predictions),
        "true_labels": np.array(all_true_labels),
        "task": task
    }

    if config["pca_enabled"]:
        results["pca_components"] = config["pca_components"]
        
    return results

def run_all_experiments(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Base Config
    base_config = {
        "mode": args.mode,
        "roi_size": args.roi_size,
        "window_radius": args.window_radii[0],
        "num_channels": args.num_channels,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "pca_components": args.pca_components,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
    }
    
    # 1. Discover Groups Once
    all_groups = get_all_groups(base_config)
    
    summary_results = []
    configurations = list(product(args.models, args.pca_settings, args.window_radii))
    
    for idx, (model, pca_enabled, window_radius) in enumerate(configurations, 1):
        print(f"\n{'='*50}\nConfig {idx}/{len(configurations)}: {model}, PCA={pca_enabled}, Radius={window_radius}\n{'='*50}")
        
        config = base_config.copy()
        config.update({
            "model": model,
            "pca_enabled": pca_enabled,
            "window_radius": window_radius,
        })
        
        # Run Tasks
        res_multi = run_kfold_cv(config, "multi", device, all_groups, n_splits=9)
        res_binary = run_kfold_cv(config, "binary", device, all_groups, n_splits=9)
        # res_binary_qs = run_kfold_cv(config, "binary_qs", device, all_groups, n_splits=9)
        res_binary_qs = {}
        # Stats
        stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "n_folds": 9,
            "groups": list(map(int, all_groups))
        }
        
        # Export
        export_results_unified(
            args.results_dir, config, 
            res_multi, res_binary, 
            res_binary_qs,
            stats, pca_enabled
        )
        
        # Log Summary
        summary_results.append({
            "model": model,
            "pca": pca_enabled,
            "window_radius": window_radius,
            "multi_acc": f"{res_multi['accuracy'][0]:.4f}±{res_multi['accuracy'][1]:.4f}",
            "multi_kappa": f"{res_multi['kappa'][0]:.4f}±{res_multi['kappa'][1]:.4f}",
            "bin_acc": f"{res_binary['accuracy'][0]:.4f}±{res_binary['accuracy'][1]:.4f}",
            "bin_kappa": f"{res_binary['kappa'][0]:.4f}±{res_binary['kappa'][1]:.4f}",
            # "qs_acc": f"{res_binary_qs['accuracy'][0]:.4f}±{res_binary_qs['accuracy'][1]:.4f}",
            # "qs_kappa": f"{res_binary_qs['kappa'][0]:.4f}±{res_binary_qs['kappa'][1]:.4f}",
        })
        
    # Save Summary CSV
    if summary_results:
        df = pd.DataFrame(summary_results)
        df.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
        print("\nSummary Table:")
        print(df.to_string(index=False))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Experiments
    parser.add_argument('--models', nargs='+', default=['gru', 'lstm', 'transformer', 'mlp'], help='Models: gru, lstm, transformer, mlp')
    parser.add_argument('--pca_settings', nargs='+', type=lambda x: x.lower()=='true', default=[True])
    parser.add_argument('--window_radii', nargs='+', type=int, default=[2])
    
    # Data params
    parser.add_argument('--mode', type=str, default='concatenated')
    parser.add_argument('--roi_size', type=int, default=41)
    parser.add_argument('--num_channels', type=int, default=22)
    parser.add_argument('--pca_components', type=int, default=50) 
    
    # Trainings
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=8)
    
    # Model Architecture
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default='../DL_results')
    
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    run_all_experiments(args)

if __name__ == "__main__":
    main()