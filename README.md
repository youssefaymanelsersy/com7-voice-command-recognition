# Voice Command Recognition System

## Workflow

### Option 1: Train (Record Samples Once)

```bash
python main.py
# Choose "1. Train"
```

- Records 20 samples per command (8 commands × 20 = 160 total samples)
- Saves all samples to `samples.json`
- Creates `model.json`
- **Do this only ONCE** - then you have your dataset

### Option 2: Rebuild (Change Logic, No Re-recording)

```bash
# Edit features.py or trainer.py with new logic
# Then:
python main.py
# Choose "2. Rebuild"
```

- Loads your saved samples from `samples.json`
- Recalculates features with your new logic
- Rebuilds the model and updates `model.json`
- **No microphone needed** - uses your existing samples

### Option 3: Run (Use Model)

```bash
python main.py
# Choose "3. Run"
```

- Recognizes spoken commands
- Press Ctrl+C to stop

## Setup

```bash
pip install numpy scipy sounddevice pyserial
```

## Script Roles

### Primary

- `main.py` - Main menu entry point (Train/Rebuild/Run)
- `audio.py` - All audio capture (COM7 serial + optional device capture)
- `trainer.py` - Training and model rebuild logic
- `recognize.py` - Command recognition logic
- `features.py` - Feature extraction logic

### Utilities

- `train.py` - Lightweight legacy wrapper for Train/Rebuild only
- `autotune.py` - Parameter search and tuned model build
- `retake_samples.py` - Re-record selected commands into `test_samples.json`
- `test.py` - Unified test menu (quick test, input devices, one-shot capture, evaluation trials)

### Data Files

- `samples.json` - Main training recordings
- `test_samples.json` - Separate autotune/test recordings
- `model.json` - Active trained model

## Troubleshooting

| Issue | Fix |
| ------- | ----- |
| "Too quiet" during training | Speak louder |
| "Clipping" during training | Speak softer |
| "Model not found" | Run Train first (option 1) |
| Need to change features? | Edit `features.py`, then Rebuild (option 2) |
| Need to change model logic? | Edit `trainer.py`, then Rebuild (option 2) |
