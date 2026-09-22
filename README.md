# Tamil LLM: Qwen2.5-1.5B-Instruct LoRA Fine-Tuning & FastAPI Serving

This project fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` for high-quality Tamil language tasks (Tamil, Tanglish, business communication, clarification, safety) using 4-bit QLoRA and exposes the model via FastAPI and an authenticated tunnel.

---

## Workspace Structure

```
kaggle_project/
├── notebook.ipynb                # Standard Jupyter Notebook (22 organized cells)
├── notebook.py                   # Clean standalone Python script for editing & diffs
├── sync_kaggle.py                # Live bidirectional sync bridge to Kaggle GPU
├── kernel-metadata.json          # Kaggle CLI metadata for remote push
├── requirements.txt              # Project dependencies
├── .gitignore                    # Prevents checking in weights, secrets, or cache
├── .github/
│   └── workflows/
│       └── kaggle_sync.yml       # Automated GitHub -> Kaggle CI/CD pipeline
└── README.md                     # Documentation
```

---

## 1. Running Locally with Jupyter

To run Jupyter locally on your machine:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch local JupyterLab or Notebook
jupyter lab
# or
jupyter notebook
```

Open `notebook.ipynb` in the browser or directly in Antigravity IDE / VS Code.

---

## 2. Real-Time Live Sync with Kaggle GPU

You can interact directly with your live Kaggle container without restarting kernels:

```powershell
# Check remote Kaggle GPU, RAM, and disk utilization
python sync_kaggle.py --status

# Execute arbitrary bash commands or Python remotely
python sync_kaggle.py --exec "!nvidia-smi"
python sync_kaggle.py --exec "import torch; print('CUDA available:', torch.cuda.is_available())"

# Push a modified local script/file to Kaggle's /kaggle/working
python sync_kaggle.py --push notebook.py

# Pull generated models or outputs from Kaggle to your local folder
python sync_kaggle.py --pull
```

---

## 3. GitHub Integration & Automated Kaggle Push (CI/CD)

### Step 1: Connect your Local Repo to GitHub
```powershell
git remote add origin https://github.com/<your-github-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

### Step 2: Configure Kaggle API Credentials in GitHub Secrets
1. Go to your Kaggle Account: **Settings -> API -> Create New Token** (downloads `kaggle.json`).
2. In your GitHub repository: Go to **Settings -> Secrets and variables -> Actions -> New repository secret**.
3. Add two secrets:
   - `KAGGLE_USERNAME`: Your Kaggle username.
   - `KAGGLE_KEY`: Your Kaggle API key (found inside `kaggle.json`).

### Step 3: Update `kernel-metadata.json`
Set the `"id"` field in `kernel-metadata.json` to your Kaggle username and notebook slug:
```json
{
  "id": "<your-username>/tamil-llm-qwen25-15b"
}
```

Whenever you run `git push origin main`, GitHub Actions will automatically push your latest notebook directly to Kaggle and trigger execution using Kaggle's cloud GPU.
