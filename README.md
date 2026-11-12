# Seatbelt Detection — local upload UI

Upload a photo → see **✅ Seatbelt worn** or **🚨 NO SEATBELT**, with probabilities. Runs on your own computer's CPU.

## 1. Train the model (once, on Google Colab)
Run `Seatbelt_Detection_CPU_Drive.ipynb` on Colab (CPU runtime is fine) using *Runtime ▸ Run all*.
When it finishes, your Google Drive has:

```
My Drive/Seatbelt_Detection/
├── seatbelt_model.zip      ← you need this
├── seatbelt_model/         (same thing, unzipped)
├── checkpoints/            (lets training resume after a disconnect)
└── split.json              (which images were train / val / test)
```

## 2. Put the model next to the app
Download **`seatbelt_model.zip`** from Drive and put it in this folder:

```
seatbelt_app/
├── app.py
├── requirements.txt
├── run_windows.bat
├── run_mac_linux.sh
└── seatbelt_model.zip      ← here (no need to unzip)
```

## 3. Run
* **Windows:** double-click `run_windows.bat`
* **Mac / Linux:** `bash run_mac_linux.sh`

The first run creates a virtual environment and installs the libraries (needs internet, takes a few minutes).
After that the browser opens **http://127.0.0.1:7860** automatically.

Manual way (any OS, Python 3.10+):
```bash
pip install -r requirements.txt
python app.py                       # or: python app.py --model "D:/Downloads/seatbelt_model.zip"
```

## Using the UI
| Control | What it does |
|---|---|
| **Upload image** | Drag a photo in (or paste from clipboard). It is checked automatically. |
| **Driver crop** | For close-ups of the driver / windshield, like the training images. Most accurate. |
| **Full scene** | YOLOv8 finds each person first, then each one is checked. Green box = belt, red = violation. The first use downloads `yolov8n.pt` (6 MB). |
| **Threshold** | Starts at the value tuned on the validation set. Lower = catches more violators but more false alarms. |

The app resizes and normalises images exactly like the notebook's test set, so it gives the same predictions as the evaluation (this is checked in Step 11 of the notebook).

## Troubleshooting
* **"Model not found"** → `seatbelt_model.zip` isn't in this folder, or use `--model path`.
* **Port in use** → `python app.py --port 8000`.
* **"Python was not found" (Windows)** → install Python 3.10+ from python.org and tick *Add python.exe to PATH*.
