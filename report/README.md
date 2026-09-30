# Report source

`report.pdf` and `report.docx` in the repo root are built from this folder. All numbers are read from the JSON files in `figures/`, so rebuilding after re-running the evaluations updates the report.

```bash
python report/render_figures.py      # needs Chrome, Pillow, and the dashboard running (python app/server.py)
cd report && npm install && cd ..
node report/make_report.js report.docx
soffice --headless --convert-to pdf report.docx
```

Add `--with-ids` to `make_report.js` for a copy with student IDs. The repo copy leaves them out.
