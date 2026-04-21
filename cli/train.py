# train.py - Legacy training interface (use trainer.py for main training functions)
# This file provides a simple entry point for the training process.

from core.trainer import train, retrain_from_saved_samples


def main():
    """Simple training interface."""
    print("\n" + "="*60)
    print("VOICE COMMAND RECOGNITION - TRAINING SYSTEM")
    print("="*60)
    print("\n1. Train new model (collect new samples)")
    print("2. Rebuild model from saved samples")
    print("3. Exit")
    print()
    
    choice = input("Choose an option (1-3): ").strip()
    
    if choice == "1":
        train()
    elif choice == "2":
        retrain_from_saved_samples()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
