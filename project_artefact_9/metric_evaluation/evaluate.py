#!/usr/bin/env python3
"""
Nutritional LLM Analysis - Evaluation Script
Computes all metrics from CSV files exported from MySQL.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CSV_DIR = "."  # Directory where CSV files are located
OUTPUT_DIR = "./results"  # Directory to save results

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LAYER 1A: EXACT MATCH
# ============================================================

def compute_exact_match(csv_file):
    """Compute exact match accuracy for ingredient names."""
    print(f"\n📊 Computing Exact Match from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {csv_file}")
        return None
    
    if 'gt_name_original' not in df.columns or 'pred_name_original' not in df.columns:
        print(f"   ⚠️ Missing required columns in {csv_file}")
        return None
    
    df['exact_match_name'] = df['gt_name_original'] == df['pred_name_original']
    df['exact_match_unit'] = df['gt_unit_original'] == df['pred_unit_original']
    df['exact_match_all'] = df['exact_match_name'] & df['exact_match_unit']
    
    results = df.groupby(['model_name', 'technique_name']).agg({
        'exact_match_name': 'mean',
        'exact_match_unit': 'mean',
        'exact_match_all': 'mean'
    }).reset_index()
    
    results.to_csv(f"{OUTPUT_DIR}/layer1a_results.csv", index=False)
    print(f"   ✅ Saved to: {OUTPUT_DIR}/layer1a_results.csv")
    return results

# ============================================================
# LAYER 1B: TEXT SIMILARITY
# ============================================================

def compute_text_similarity(csv_file):
    """Compute fuzzy match, BLEU, and ROUGE scores."""
    print(f"\n📊 Computing Text Similarity from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {csv_file}")
        return None
    
    if 'gt_name_original' not in df.columns or 'pred_name_original' not in df.columns:
        print(f"   ⚠️ Missing required columns in {csv_file}")
        return None
    
    try:
        from rapidfuzz import fuzz
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        smoothie = SmoothingFunction().method4
        
        for idx, row in df.iterrows():
            gt = str(row['gt_name_original'])
            pred = str(row['pred_name_original'])
            
            df.loc[idx, 'fuzzy_score'] = fuzz.token_sort_ratio(gt, pred)
            
            reference = [gt.split()]
            candidate = pred.split()
            
            df.loc[idx, 'bleu_1'] = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            df.loc[idx, 'bleu_2'] = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
            
            rouge_scores = scorer.score(gt, pred)
            df.loc[idx, 'rouge_1'] = rouge_scores['rouge1'].fmeasure
            df.loc[idx, 'rouge_l'] = rouge_scores['rougeL'].fmeasure
        
        results = df.groupby(['model_name', 'technique_name']).agg({
            'fuzzy_score': 'mean',
            'bleu_1': 'mean',
            'bleu_2': 'mean',
            'rouge_1': 'mean',
            'rouge_l': 'mean'
        }).reset_index()
        
        results.to_csv(f"{OUTPUT_DIR}/layer1b_results.csv", index=False)
        print(f"   ✅ Saved to: {OUTPUT_DIR}/layer1b_results.csv")
        return results
        
    except ImportError as e:
        print(f"   ⚠️ Missing package: {e}")
        return None

# ============================================================
# LAYER 2A: NUMERIC ACCURACY (MAE, MAPE)
# ============================================================

def compute_numeric_accuracy(csv_file):
    """Compute MAE and MAPE for numeric values."""
    print(f"\n📊 Computing Numeric Accuracy from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {csv_file}")
        return None
    
    results = df.groupby(['model_name', 'technique_name']).apply(
        lambda g: pd.Series({
            'mae_quantity': np.mean(np.abs(g['gt_quantity_value'] - g['pred_quantity_value'])),
            'mape_quantity': np.mean(np.where(g['gt_quantity_value'] != 0, 
                                             np.abs(g['gt_quantity_value'] - g['pred_quantity_value']) / np.abs(g['gt_quantity_value']) * 100, 0)),
            'mae_weight': np.mean(np.abs(g['gt_weight_g'] - g['pred_weight_g'])),
            'mape_weight': np.mean(np.where(g['gt_weight_g'] != 0,
                                           np.abs(g['gt_weight_g'] - g['pred_weight_g']) / np.abs(g['gt_weight_g']) * 100, 0))
        })
    ).reset_index()
    
    results.to_csv(f"{OUTPUT_DIR}/layer2a_results.csv", index=False)
    print(f"   ✅ Saved to: {OUTPUT_DIR}/layer2a_results.csv")
    return results

# ============================================================
# LAYER 2B: NUMERIC NUTRITION (MAE, MAPE, PEARSON)
# ============================================================

def compute_numeric_nutrition(csv_file):
    """Compute MAE, MAPE, and Pearson for nutrition values."""
    print(f"\n📊 Computing Numeric Nutrition from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {csv_file}")
        return None
    
    # Check required columns
    nutrition_fields = ['energy_kcal', 'protein_g', 'fat_g', 'carbohydrate_g']
    results_data = []
    
    for field in nutrition_fields:
        gt_col = f'gt_{field}'
        pred_col = f'pred_{field}'
        
        if gt_col not in df.columns or pred_col not in df.columns:
            print(f"   ⚠️ Missing columns for {field}")
            continue
        
        # Group by model and technique
        grouped = df.groupby(['model_name', 'technique_name'])
        
        for name, group in grouped:
            model_name, technique_name = name
            gt_vals = group[gt_col].values
            pred_vals = group[pred_col].values
            
            # MAE
            mae = np.mean(np.abs(gt_vals - pred_vals))
            
            # MAPE
            mape = np.mean(np.where(gt_vals != 0, 
                                   np.abs(gt_vals - pred_vals) / np.abs(gt_vals) * 100, 0))
            
            # Pearson correlation
            if len(gt_vals) > 1 and np.std(gt_vals) > 0 and np.std(pred_vals) > 0:
                pearson = np.corrcoef(gt_vals, pred_vals)[0, 1]
            else:
                pearson = np.nan
            
            results_data.append({
                'model_name': model_name,
                'technique_name': technique_name,
                'nutrient': field,
                'mae': mae,
                'mape': mape,
                'pearson': pearson
            })
    
    if results_data:
        results = pd.DataFrame(results_data)
        results.to_csv(f"{OUTPUT_DIR}/layer2b_results.csv", index=False)
        print(f"   ✅ Saved to: {OUTPUT_DIR}/layer2b_results.csv")
    else:
        print(f"   ⚠️ No data processed for Layer 2B")
    
    return results_data

# ============================================================
# LAYER 2C: RECIPE TOTALS
# ============================================================

def compute_recipe_totals(csv_file):
    """Compute recipe-level totals comparison."""
    print(f"\n📊 Computing Recipe Totals from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {csv_file}")
        return None
    
    # Check if predicted columns exist and are not all NULL
    pred_cols = ['pred_total_energy_kcal', 'pred_total_protein_g', 'pred_total_fat_g', 'pred_total_carbohydrate_g']
    has_pred_data = all(col in df.columns and not df[col].isna().all() for col in pred_cols)
    
    if not has_pred_data:
        print(f"   ⚠️ No predicted data found in {csv_file}. Skipping Layer 2C.")
        return None
    
    # Group by model and technique
    results = df.groupby(['model_name', 'technique_name']).apply(
        lambda g: pd.Series({
            'gt_avg_calories': g['gt_total_energy_kcal'].mean(),
            'pred_avg_calories': g['pred_total_energy_kcal'].mean(),
            'mae_calories': np.mean(np.abs(g['gt_total_energy_kcal'] - g['pred_total_energy_kcal'])),
            'mape_calories': np.mean(np.where(g['gt_total_energy_kcal'] != 0,
                                             np.abs(g['gt_total_energy_kcal'] - g['pred_total_energy_kcal']) / np.abs(g['gt_total_energy_kcal']) * 100, 0)),
            'gt_avg_protein': g['gt_total_protein_g'].mean(),
            'pred_avg_protein': g['pred_total_protein_g'].mean(),
            'mae_protein': np.mean(np.abs(g['gt_total_protein_g'] - g['pred_total_protein_g'])),
            'gt_avg_fat': g['gt_total_fat_g'].mean(),
            'pred_avg_fat': g['pred_total_fat_g'].mean(),
            'mae_fat': np.mean(np.abs(g['gt_total_fat_g'] - g['pred_total_fat_g'])),
            'gt_avg_carbs': g['gt_total_carbohydrate_g'].mean(),
            'pred_avg_carbs': g['pred_total_carbohydrate_g'].mean(),
            'mae_carbs': np.mean(np.abs(g['gt_total_carbohydrate_g'] - g['pred_total_carbohydrate_g']))
        })
    ).reset_index()
    
    results.to_csv(f"{OUTPUT_DIR}/layer2c_results.csv", index=False)
    print(f"   ✅ Saved to: {OUTPUT_DIR}/layer2c_results.csv")
    return results

# ============================================================
# LAYER 3A: JSON VALIDITY
# ============================================================

def compute_output_quality(validity_csv, hallucination_csv):
    """Compute JSON validity and hallucination rates."""
    print(f"\n📊 Computing Output Quality...")
    
    try:
        df_valid = pd.read_csv(validity_csv)
        df_valid.to_csv(f"{OUTPUT_DIR}/layer3a_results.csv", index=False)
        print(f"   ✅ Saved JSON validity to: {OUTPUT_DIR}/layer3a_results.csv")
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {validity_csv}")
    
    try:
        df_halluc = pd.read_csv(hallucination_csv)
        if 'is_hallucinated' in df_halluc.columns:
            results = df_halluc.groupby(['model_name', 'technique_name']).agg({
                'is_hallucinated': lambda x: np.mean(pd.to_numeric(x, errors='coerce'))
            }).reset_index()
            results.columns = ['model_name', 'technique_name', 'hallucination_rate']
            results.to_csv(f"{OUTPUT_DIR}/layer3b_results.csv", index=False)
            print(f"   ✅ Saved hallucination rate to: {OUTPUT_DIR}/layer3b_results.csv")
        else:
            print(f"   ⚠️ 'is_hallucinated' column not found in {hallucination_csv}")
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {hallucination_csv}")

# ============================================================
# LAYER 3C: INGREDIENT METRICS
# ============================================================

def compute_ingredient_metrics(csv_file):
    """Compute Precision, Recall, and F1 for ingredient detection."""
    print(f"\n📊 Computing Ingredient Metrics from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {csv_file}")
        return None
    
    if 'true_positives' not in df.columns or 'false_positives' not in df.columns:
        print(f"   ⚠️ Missing 'true_positives' or 'false_positives' columns in {csv_file}")
        print(f"   Current columns: {df.columns.tolist()}")
        return None
    
    def compute_f1(row):
        tp = row['true_positives']
        fp = row['false_positives']
        fn = row['gt_ingredient_count'] - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return pd.Series({'precision': precision, 'recall': recall, 'f1': f1})
    
    df[['precision', 'recall', 'f1']] = df.apply(compute_f1, axis=1)
    
    results = df.groupby(['model_name', 'technique_name']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean'
    }).reset_index()
    
    results.to_csv(f"{OUTPUT_DIR}/layer3c_results.csv", index=False)
    print(f"   ✅ Saved to: {OUTPUT_DIR}/layer3c_results.csv")
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     🍳 Nutritional LLM Analysis - Evaluation Script       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    csv_files = {
        'layer1a': 'layer1a_exact_match.csv',
        'layer1b': 'layer1b_text_similarity.csv',
        'layer2a': 'layer2a_numeric_quantity.csv',
        'layer2b': 'layer2b_numeric_nutrition.csv',
        'layer2c': 'layer2c_nutrition_totals.csv',
        'layer3a': 'layer3a_json_validity.csv',
        'layer3b': 'layer3b_hallucination.csv',
        'layer3c': 'layer3c_ingredient_detection.csv'
    }
    
    # Run evaluations
    if os.path.exists(csv_files['layer1a']):
        compute_exact_match(csv_files['layer1a'])
    else:
        print(f"⚠️ File not found: {csv_files['layer1a']}")
    
    if os.path.exists(csv_files['layer1b']):
        compute_text_similarity(csv_files['layer1b'])
    else:
        print(f"⚠️ File not found: {csv_files['layer1b']}")
    
    if os.path.exists(csv_files['layer2a']):
        compute_numeric_accuracy(csv_files['layer2a'])
    else:
        print(f"⚠️ File not found: {csv_files['layer2a']}")
    
    if os.path.exists(csv_files['layer2b']):
        compute_numeric_nutrition(csv_files['layer2b'])
    else:
        print(f"⚠️ File not found: {csv_files['layer2b']}")
    
    if os.path.exists(csv_files['layer2c']):
        compute_recipe_totals(csv_files['layer2c'])
    else:
        print(f"⚠️ File not found: {csv_files['layer2c']}")
    
    if os.path.exists(csv_files['layer3a']) and os.path.exists(csv_files['layer3b']):
        compute_output_quality(csv_files['layer3a'], csv_files['layer3b'])
    else:
        print(f"⚠️ File(s) not found: {csv_files['layer3a']} or {csv_files['layer3b']}")
    
    if os.path.exists(csv_files['layer3c']):
        compute_ingredient_metrics(csv_files['layer3c'])
    else:
        print(f"⚠️ File not found: {csv_files['layer3c']}")
    
    print("\n✅ Evaluation Complete!")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()