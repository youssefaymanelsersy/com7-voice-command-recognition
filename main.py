import json
from audio import record_audio, using_serial_input
from app_config import (
    QUIET_THRESHOLD,
    CLIPPING_THRESHOLD,
    MSG_SERIAL_MODE,
    MSG_TOO_QUIET,
    MSG_CLIPPING,
    MSG_INVALID_CHOICE,
)
from recognize import recognize
from trainer import train, retrain_from_saved_samples


# ============================================================================
# WORKFLOW
# ============================================================================
# Option 1 (Train): 
#   Record 20 samples per command, save to samples.json
#   Do this ONCE, then never touch it again
#
# Option 2 (Rebuild):
#   Modify features.py or trainer.py with new feature logic
#   Use this to rebuild model.json from the saved samples
#   NO RE-RECORDING NEEDED
#
# Option 3 (Run):
#   Use the model to recognize commands
# ============================================================================


def run_recognition():
    """Run voice command recognition."""
    try:
        with open("model.json", "r") as f:
            model = json.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Train first.")
        return

    serial_mode = using_serial_input()
    print("Listening... (Ctrl+C to stop)\n")
    if serial_mode:
        print(f"{MSG_SERIAL_MODE}\n")

    try:
        while True:
            if not serial_mode:
                input("Press Enter to speak: ")
            _, audio, max_amp = record_audio()
            
            if max_amp < QUIET_THRESHOLD:
                print(f"{MSG_TOO_QUIET}\n")
                continue

            if (not serial_mode) and max_amp > CLIPPING_THRESHOLD:
                print(f"{MSG_CLIPPING}\n")
                continue
            
            cmd = recognize(audio, model)
            print(f"Result: {cmd}\n")
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*50)
        print("1. Train (record samples ONCE)")
        print("2. Rebuild (use saved samples with new logic)")
        print("3. Run (recognize commands)")
        print("4. Exit")
        print("="*50)
        
        choice = input("Choice (1-4): ").strip()
        
        if choice == "1":
            train()
        elif choice == "2":
            retrain_from_saved_samples()
        elif choice == "3":
            run_recognition()
        elif choice == "4":
            break
        else:
            print(MSG_INVALID_CHOICE)


if __name__ == "__main__":
    main()
