import json
import sys

from cli.autotune import main as autotune_main
from cli.build_zcr_energy_model import main as build_zcr_energy_main
from cli.evaluate import main as evaluate_main
from cli.play_processed_signal import main as play_processed_signal_main
from cli.retake_samples import main as retake_samples_main
from cli.retake_training_samples import main as retake_training_samples_main
from cli.test import main as test_menu_main
from core.audio import record_audio, using_serial_input
from core.app_config import (
    QUIET_THRESHOLD,
    CLIPPING_THRESHOLD,
    SERIAL_PORT,
    MODEL_FILE,
    PROCESSED_SIGNAL_TX_PORT,
    PROCESSED_SIGNAL_TX_BAUDRATE,
    MSG_SERIAL_MODE,
    MSG_TOO_QUIET,
    MSG_CLIPPING,
    MSG_INVALID_CHOICE,
)
from core.features import extract_frame_features
from core.processed_signal_io import transmit_processed_signal
from core.recognize import recognize_features
from core.trainer import train, retrain_from_saved_samples


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
        with open(MODEL_FILE, "r") as f:
            model = json.load(f)
    except FileNotFoundError:
        print(f"❌ Model not found: {MODEL_FILE}. Train first.")
        return

    serial_mode = using_serial_input()
    tx_enabled = bool(PROCESSED_SIGNAL_TX_PORT)
    print("Listening... (Ctrl+C to stop)\n")
    if serial_mode:
        print(f"{MSG_SERIAL_MODE}\n")
    if tx_enabled:
        print(
            f"Processed signal TX enabled on {PROCESSED_SIGNAL_TX_PORT} @ {PROCESSED_SIGNAL_TX_BAUDRATE}."
        )

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
            
            frame_features = extract_frame_features(audio)
            cmd = recognize_features(frame_features, model)

            # Send only after full processing is complete and buffered.
            processed_signal = frame_features.get("processed_signal")
            if tx_enabled and processed_signal is not None and len(processed_signal) > 0:
                transmit_processed_signal(
                    processed_signal,
                    PROCESSED_SIGNAL_TX_PORT,
                    PROCESSED_SIGNAL_TX_BAUDRATE,
                )

            print(f"Result: {cmd}\n")
    except KeyboardInterrupt:
        print("\nStopped.")


def _run_cli_main(target_main):
    """Run a sub-CLI from the main hub without inheriting unrelated argv."""
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        target_main()
    finally:
        sys.argv = old_argv


def _run_cli_main_with_args(target_main, args):
    """Run a sub-CLI with explicit argv values."""
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *args]
        target_main()
    finally:
        sys.argv = old_argv


def main():
    """Main menu."""
    while True:
        print("\n" + "="*50)
        print("1. Train (record samples ONCE, frame-based features)")
        print("2. Rebuild (from saved extracted features)")
        print("3. Run (recognize commands + optional processed-signal TX)")
        print("4. Autotune model")
        print("5. Evaluate model offline")
        print("6. Retake selected TEST samples")
        print("7. Retake selected TRAINING samples")
        print("8. Build ZCR+energy model")
        print("9. Test tools menu")
        print("10. Play processed serial signal")
        print("11. Exit")
        print("="*50)
        
        try:
            choice = input("Choice (1-11): ").strip()
        except EOFError:
            print("\nExiting.")
            break
        
        if choice == "1":
            train()
        elif choice == "2":
            retrain_from_saved_samples()
        elif choice == "3":
            run_recognition()
        elif choice == "4":
            _run_cli_main(autotune_main)
        elif choice == "5":
            _run_cli_main(evaluate_main)
        elif choice == "6":
            _run_cli_main(retake_samples_main)
        elif choice == "7":
            _run_cli_main(retake_training_samples_main)
        elif choice == "8":
            _run_cli_main(build_zcr_energy_main)
        elif choice == "9":
            _run_cli_main(test_menu_main)
        elif choice == "10":
            default_port = PROCESSED_SIGNAL_TX_PORT or SERIAL_PORT
            port = input(f"Serial port [{default_port}]: ").strip() or default_port
            sample_rate = input("Sample rate [8000]: ").strip() or "8000"
            _run_cli_main_with_args(
                play_processed_signal_main,
                [
                    "--port",
                    port,
                    "--baudrate",
                    str(PROCESSED_SIGNAL_TX_BAUDRATE),
                    "--sample-rate",
                    sample_rate,
                ],
            )
        elif choice == "11":
            break
        else:
            print(MSG_INVALID_CHOICE)


if __name__ == "__main__":
    main()
