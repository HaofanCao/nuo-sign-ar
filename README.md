# AR Hand-Sign Demo for Nuo Opera

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D4)
![License](https://img.shields.io/badge/License-MIT-green)

Realtime AR hand-sign recognition demo for Nuo Opera scenes.  
This repository focuses on runnable demo implementation and reproducible scripts.

## Demo Preview

| AR Interface Preview |
| --- |
| ![AR UI preview](outputs/ar_ui_preview.png) |

## Features

- Realtime hand landmark detection with MediaPipe.
- Rule-based hand-sign recognition with confidence scoring.
- AR overlay rendering on live webcam frames.
- Screenshot and recording capture during runtime.
- Parameter benchmark utility for smoothing/threshold/margin comparison.

## Project Structure

```text
.
|-- .gitignore
|-- LICENSE
|-- app.py
|-- hand_sign_ar/
|   |-- recognizer.py
|   |-- overlay.py
|   |-- benchmark.py
|   `-- __init__.py
|-- scripts/
|   |-- run_parameter_benchmark.py
|   `-- generate_ui_preview.py
|-- outputs/
|-- requirements.txt
|-- setup_venv.ps1
|-- setup_venv.bat
`-- run_demo.bat
```

## Requirements

- Windows 10/11 (PowerShell commands are provided)
- Python 3.10+
- Webcam

## Quick Start

1. Create and initialize a virtual environment:

```powershell
powershell -ExecutionPolicy Bypass -File setup_venv.ps1
```

2. Run the demo:

```powershell
.\.venv\Scripts\python.exe app.py --camera 0 --backend tasks
```

3. Or use the one-click launcher:

```powershell
.\run_demo.bat
```

## CLI Usage

```powershell
.\.venv\Scripts\python.exe app.py `
  --camera 0 `
  --width 1280 `
  --height 720 `
  --min-detect 0.65 `
  --min-track 0.60 `
  --smoothing 5 `
  --threshold 0.55 `
  --margin 0.06 `
  --output outputs `
  --backend auto
```

Key options:

- `--camera`: camera index
- `--output`: output root path for captures and recordings
- `--backend`: `auto`, `tasks`, or `solutions`
- `--smoothing`, `--threshold`, `--margin`: recognition behavior controls

## Keyboard Controls

- `Q` or `Esc`: quit
- `S`: save screenshot
- `R`: start or stop recording

## Output Files

- Default output root is `outputs/` under this folder.
- Screenshots: `outputs/screenshots/`
- Recordings: `outputs/recordings/`
- If `--output` is an absolute path (for example `C:/AR_demo_outputs`), files are written there.
- MediaPipe task model defaults to `C:/AR_demo_models/hand_landmarker.task` (can be overridden by `--task-model`).
- Sample preview images for this README are stored in `outputs/`.

## Utility Scripts

Generate parameter comparison results:

```powershell
.\.venv\Scripts\python.exe scripts/run_parameter_benchmark.py
```

Generate simulated UI preview image:

```powershell
.\.venv\Scripts\python.exe scripts/generate_ui_preview.py
```

## Troubleshooting

- Error: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
  - Cause: mixed global/conda package versions.
  - Fix: use this project's `.venv` only and reinstall dependencies.
- Error: `module 'mediapipe' has no attribute 'solutions'`
  - Cause: MediaPipe API differences across versions.
  - Fix: use `--backend tasks` or `--backend auto`.

## Contributing

1. Create a feature branch.
2. Keep changes scoped and testable.
3. Open a pull request with summary, steps to run, and expected results.

## License

This project is licensed under the MIT License.  
See [LICENSE](LICENSE) for details.

