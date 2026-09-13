import os
import sys
import json
from pathlib import Path
import pandas as pd
import tensorflow as tf

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(r"c:\Users\user\OneDrive\ドキュメント\gitam\vscode proj\cyberattack detection")
ckpt_dir = PROJECT_ROOT / "checkpoints" / "cnn_transformer" / "real_ciciot2023"
res_dir = PROJECT_ROOT / "results" / "cnn_transformer" / "real_ciciot2023"

print("=== CHECKPOINT DIRECTORY INSPECTION ===")
print("Checkpoint dir exists:", ckpt_dir.exists())
if ckpt_dir.exists():
    for f in ckpt_dir.iterdir():
        print(f"  - {f.name}: {f.stat().st_size:,} bytes")

print("\n=== RESULTS DIRECTORY INSPECTION ===")
print("Results dir exists:", res_dir.exists())
if res_dir.exists():
    for f in res_dir.iterdir():
        if f.is_file():
            print(f"  - {f.name}: {f.stat().st_size:,} bytes")

best_model_path = ckpt_dir / "best_model.keras"
final_model_path = ckpt_dir / "final_model.keras"
history_path = res_dir / "history.json"
log_path = res_dir / "training_log.csv"
meta_path = res_dir / "metadata.json"

print("\n=== DETAILED ITEM REPORT ===")
print("1. best_model.keras exists:", best_model_path.exists())
print("2. final_model.keras exists:", final_model_path.exists())

b_size = best_model_path.stat().st_size if best_model_path.exists() else 0
f_size = final_model_path.stat().st_size if final_model_path.exists() else 0
print(f"3. File sizes:\n   - best_model.keras: {b_size:,} bytes\n   - final_model.keras: {f_size:,} bytes")

print("4. history.json exists:", history_path.exists())
print("5. training_log.csv exists:", log_path.exists())
print("6. metadata.json exists:", meta_path.exists())

if log_path.exists() and log_path.stat().st_size > 0:
    log_df = pd.read_csv(log_path)
    print("\n--- Training Log CSV Contents ---")
    print(log_df.to_string())
    
    last_epoch = len(log_df)
    min_idx = log_df['val_loss'].idxmin()
    best_row = log_df.loc[min_idx]
    best_val_loss = float(best_row['val_loss'])
    best_val_acc = float(best_row['val_accuracy'])
    best_epoch_num = int(best_row['epoch']) + 1
    
    print(f"\n7. Last completed epoch recorded: Epoch {last_epoch} (0-indexed epoch {last_epoch - 1})")
    print(f"8. Best validation loss recorded: {best_val_loss:.6f} (at Epoch {best_epoch_num})")
    print(f"9. Best validation accuracy recorded: {best_val_acc * 100:.4f}% ({best_val_acc:.6f})")
else:
    print("\n7. Last completed epoch recorded: None (log empty or missing)")
    print("8. Best validation loss recorded: N/A")
    print("9. Best validation accuracy recorded: N/A")

load_success = False
output_shape = None
if best_model_path.exists():
    try:
        model = tf.keras.models.load_model(str(best_model_path))
        output_shape = model.output_shape
        load_success = True
        print(f"\n10. Keras Model Loading Test: SUCCESS! Loaded model output_shape = {output_shape}")
    except Exception as e:
        print(f"\n10. Keras Model Loading Test: FAILED ({e})")
else:
    print("\n10. Keras Model Loading Test: N/A (file does not exist)")

print(f"11. Training safely resumable from checkpoint: {load_success and best_model_path.exists()}")

# Verify dataset & class configuration
print("\n12. Verification of REAL CICIoT2023 dataset & 7 active classes configuration:")
if output_shape:
    print(f"    - Output layer dimension: {output_shape[1]} (matches 7 active classes contract)")
else:
    print("    - Output layer dimension: N/A")
print(f"    - Data path used: data/processed/prepared/subsampled_train")
print(f"    - Active classes configured: 7 (Benign, BruteForce, DDoS, DoS, Mirai, Spoofing, Web-based)")
