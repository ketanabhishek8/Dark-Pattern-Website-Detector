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

## How the model is trained and checked

`app/train.py` trains one DistilBERT model with 6 classes (not a dark pattern, Scarcity, Urgency, Social Proof, Misdirection, Other) on:
- the dataset's training split, split **by website** so the test set only contains sites the model hasn't seen (`app/data.py`)
- Indian-style template lines (`app/india_data.py`)
- 171 hand-labelled lines **mined from 17 real product pages** where the first model made mistakes (`app/real_pages/mined.json`)

Two ways to check it:
- `python app/evaluate.py`: scores on the dataset's test split and on 32 held-out Indian-style lines.
- `python app/eval_real_pages.py`: runs the whole pipeline (page, then lines, then model) on hand-labelled product pages: 5 dev and 5 test pages from Amazon.in and Snapdeal, plus 6 pages from **sites never used in training** (`app/real_pages/labels.json`). The page snapshots stay local (`app/real_pages/html/`, not in git).

| Real product pages (shipped threshold 0.97) | Test pages (Amazon/Snapdeal) | Unseen sites |
|---|---|---|
| First model (2-class) | F1 0.50, 1.8 false alarms per page | F1 0.22, 2.2 false alarms per page |
| Current model (6-class, mined data) | F1 1.00, no false alarms | F1 0.80, no false alarms |

The threshold was first chosen on the dev pages (step 3) and kept when the model was retrained. That decision was made after seeing the unseen-site results, so treat the unseen F1 as optimistic. The unseen and test sets are small (3 and 9 dark lines).

## Deploy to Hugging Face Spaces

The app runs as a Docker Space (see `Dockerfile`). When hosted, it refuses links to private network addresses.

```bash
source .venv/bin/activate
pip install huggingface_hub
hf auth login                   # paste a token with write access from huggingface.co/settings/tokens
python deploy/deploy_space.py   # creates or updates the Space and uploads the app with the model
```
