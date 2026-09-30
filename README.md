# Dark Pattern Detector

Abhishek Joshi and Shaurya Ghorpade, Artificial Intelligence course.

| File | What it is |
|---|---|
| `report.pdf` | The 3-page project report (`report.docx` is the editable Word version) |
| `dark_pattern_detector.ipynb` | Experiments: TF-IDF baselines vs DistilBERT (run on Google Colab with a T4 GPU) |
| `app/` | The web dashboard: paste a product page link and see its dark patterns |
| `figures/` | Charts and results used in the report |

## Run the dashboard

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
python app/train.py      # about 3 minutes on an M-series Mac, saves the model to app/model/
python app/server.py     # then open http://localhost:8000
```

The trained model (`app/model/`, about 256 MB) is too big for GitHub, so it isn't in the repo. Run `train.py` once after cloning to create it. It downloads the dataset, trains DistilBERT and saves the model. It works without a GPU, just more slowly.

Open http://localhost:8000/?demo to scan the built-in sample shop straight away. That is the safest option for a live demo, because some real sites block automated requests. For those sites, use **Paste text** instead.

To re-check the numbers in Table 2 of the report: `python app/evaluate.py`

## Deploy to Hugging Face Spaces

The app runs as a Docker Space (see `Dockerfile`). When hosted, it refuses links to private network addresses.

```bash
source .venv/bin/activate
pip install huggingface_hub
hf auth login                   # paste a token with write access from huggingface.co/settings/tokens
python deploy/deploy_space.py   # creates or updates the Space and uploads the app with the model
```
