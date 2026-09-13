import sys
import time
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\user\OneDrive\ドキュメント\gitam\vscode proj\cyberattack detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix console stdout encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

from src.training.train import run_training
from src.evaluation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

def main():
    LOGGER.info("==========================================================")
    LOGGER.info("STARTING FRESH 50-EPOCH REAL CICIoT2023 CNN-TRANSFORMER BASELINE EXPERIMENT")
    LOGGER.info("==========================================================")
    
    data_dir = PROJECT_ROOT / "data" / "processed" / "prepared" / "subsampled_train"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "cnn_transformer" / "real_ciciot2023" / "fresh_50_epochs"
    results_dir = PROJECT_ROOT / "results" / "cnn_transformer" / "real_ciciot2023" / "fresh_50_epochs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Training Parameters:")
    LOGGER.info("  Data directory: %s", data_dir)
    LOGGER.info("  Checkpoint directory: %s", checkpoint_dir)
    LOGGER.info("  Results directory: %s", results_dir)
    LOGGER.info("  Epochs: 50")
    LOGGER.info("  Batch size: 1024")
    LOGGER.info("  Learning rate: 0.00005")
    LOGGER.info("  Seed: 42")
    LOGGER.info("  Active classes: 7")

    # Step 1: Run Fresh Training (50 Epochs)
    train_start = time.time()
    train_results = run_training(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=results_dir,
        epochs=50,
        batch_size=1024,
        learning_rate=0.00005,
        seed=42,
        early_stopping=False,
    )
    train_duration = time.time() - train_start
    LOGGER.info("Training finished in %.2f seconds.", train_duration)

    # Step 2: Run Evaluation on Untouched Natural Test Set
    best_checkpoint = checkpoint_dir / "best_model.keras"
    test_dir = data_dir / "test"
    LOGGER.info("Evaluating BEST checkpoint (%s) on untouched test set (%s)...", best_checkpoint, test_dir)
    
    eval_results = evaluate_model(
        checkpoint_path=best_checkpoint,
        test_dir=test_dir,
        output_dir=results_dir,
        batch_size=1024,
    )

    LOGGER.info("==========================================================")
    LOGGER.info("FRESH REAL CICIoT2023 CNN-TRANSFORMER EXPERIMENT COMPLETE")
    LOGGER.info("==========================================================")

if __name__ == "__main__":
    main()
