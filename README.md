# Voice Command Recognition System

## Project Layout

```text
.
├── core/                 # Reusable library modules (audio, features, model logic)
├── cli/                  # Full CLI workflow implementations
├── data/                 # Recorded datasets
├── models/               # Trained model JSON files
├── include/              # Generated C headers for embedded side
└── main.py               # Single user-facing entry point
```

Use `python main.py` as the primary interface.

`python main.py` is now the unified interface for most workflows:

- Train/Rebuild/Run
- Autotune
- Offline evaluate
- Retake selected TEST samples
- Retake selected TRAINING samples
- Build ZCR+energy model
- Test tools menu

### Option 1: Train (Record Samples Once)

```bash
.venv/bin/python main.py
# Choose "1. Train"
```

- Records 20 samples per command (8 commands × 20 = 160 total samples)
- Saves all samples to `data/samples.json`
- Creates `models/model.json`
- **Do this only ONCE** - then you have your dataset

### Option 2: Rebuild (Change Logic, No Re-recording)

```bash
# Edit core/features.py or core/trainer.py with new logic
# Then:
.venv/bin/python main.py
# Choose "2. Rebuild"
```

- Loads your saved samples from `data/samples.json`
- Recalculates features with your new logic
- Rebuilds the model and updates `models/model.json`
- **No microphone needed** - uses your existing samples

### Option 3: Run (Use Model)

```bash
VOICE_MODEL_FILE=models/model_zcr_energy.json .venv/bin/python main.py
# Choose "3. Run"
```

- Recognizes spoken commands
- Press Ctrl+C to stop

### Option 4: Evaluate Offline (No Board/Microphone)

```bash
.venv/bin/python main.py
# Choose "5. Evaluate model offline"
```

- Uses saved feature samples only
- Prints overall accuracy, per-command accuracy, and a confusion matrix
- Does not import or require audio hardware

### Option 5: Tune Offline

```bash
.venv/bin/python main.py
# Choose "4. Autotune model"
```

- Keeps `data/samples.json` as the training set
- Uses `data/test_samples.json` as validation data
- Updates `models/model.json` and regenerates `include/voice_model.h`
- For the embedded ZCR/energy model, use `main.py` option 8

## Setup

```bash
python3 -m pip install numpy scipy sounddevice pyserial
```

Offline evaluation/tuning only needs `numpy` and `scipy`.

On systems that block system-wide pip installs, use the repo-local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Hardware Configuration

Runtime settings can be overridden without editing code:

```bash
VOICE_SERIAL_PORT=COM7
VOICE_SERIAL_BAUDRATE=230400
VOICE_AUDIO_SAMPLE_RATE=8000
VOICE_RECORD_DURATION_SEC=1
VOICE_DEVICE_ID=2
VOICE_MODEL_FILE=models/model.json
VOICE_PROCESSED_SIGNAL_TX_PORT=COM8
VOICE_PROCESSED_SIGNAL_TX_BAUDRATE=115200
```

Set `VOICE_SERIAL_PORT=""` to use a microphone input instead of serial mode.
Set `VOICE_MODEL_FILE=models/model_zcr_energy.json` to run the ZCR/energy-only model.
Set `VOICE_PROCESSED_SIGNAL_TX_PORT=COMx` to stream processed samples as `START`/`END` framed integers.

## Processed Signal Playback (Serial)

When `VOICE_PROCESSED_SIGNAL_TX_PORT` is set, option 3 in `main.py` sends the processed signal buffer only after full processing completes.

Frame format:

```text
START
sample_1
sample_2
...
sample_N
END
```

PC-side playback:

```bash
python -m cli.play_processed_signal --port COM8 --baudrate 115200 --sample-rate 8000
```

The player waits for `END` before playback.

## ZCR + Energy Model

The embedded project should use the two-feature model that ignores length and spectral centroid:

```bash
.venv/bin/python main.py
# Choose "8. Build ZCR+energy model"

.venv/bin/python main.py
# Choose "5. Evaluate model offline"

VOICE_MODEL_FILE=models/model_zcr_energy.json .venv/bin/python main.py
# Choose "3. Run"
```

- `models/model.json` / `include/voice_model.h` remain the full four-feature model
- `models/model_zcr_energy.json` / `include/voice_model_zcr_energy.h` use only `zcr` and `energy`
- `include/samplesZteAndZcr.h` remains the legacy ZCR/energy sample export
- Current validation: `67/80` correct (`83.8%`) with `5/80` unknown (`6.2%`)

## Script Roles

### Primary

- `main.py` - Unified entry point and menu
- `core/audio.py` - All audio capture (COM7 serial + optional device capture)
- `core/trainer.py` - Training and model rebuild logic
- `core/recognize.py` - Command recognition logic
- `core/features.py` - Feature extraction logic

Frame extraction uses fixed non-overlapping windows per recording:

- 8000 total samples per recording
- 32 frames
- 250 samples per frame
- Per-recording outputs: `zcr_features[32]` and `ste_features[32]`
- Per-word outputs (after 20 recordings): `zcr_avg[32]` and `ste_avg[32]`

### Utilities

- `cli/autotune.py` - Parameter search/tuning workflow
- `cli/evaluate.py` - Offline model evaluation
- `cli/build_zcr_energy_model.py` - Build/tune ZCR+energy-only model
- `cli/retake_samples.py` - Re-record selected TEST samples (`data/test_samples.json`)
- `cli/retake_training_samples.py` - Re-record selected TRAINING samples (`data/samples.json`)
- `cli/test.py` - Unified test tools menu
- `cli/migrate_sample_rate.py` - Rescale saved spectral-centroid features after sample-rate changes

### Data Files

- `data/samples.json` - Main training recordings
- `data/frame_feature_averages.json` - Per-word frame averages (`zcr_avg` and `ste_avg`, each size 32)
- `data/test_samples.json` - Separate autotune/test recordings
- `models/model.json` - Active trained model
- `models/model_zcr_energy.json` - Separate model using only ZCR and energy
- `include/voice_model.h` - Full C export of active model parameters and feature vectors
- `include/voice_model_zcr_energy.h` - C export for the ZCR/energy-only model
- `include/samplesZteAndZcr.h` - Legacy C export containing only ZCR/energy samples

## Troubleshooting

| Issue | Fix |
| ------- | ----- |
| "Too quiet" during training | Speak louder |
| "Clipping" during training | Speak softer |
| "Model not found" | Run Train first (option 1) |
| Need to change features? | Edit `core/features.py`, then Rebuild (option 2) |
| Need to change model logic? | Edit `core/trainer.py`, then Rebuild (option 2) |
| Need a quick accuracy check? | Run `main.py` option 5 |
| Python works but embedded does not match? | Regenerate `include/voice_model.h` after tuning/rebuilding |
