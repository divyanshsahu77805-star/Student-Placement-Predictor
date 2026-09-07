# Placement Board

A small web app that predicts whether a student is likely to get placed,
based on the model I trained in `MDS_Draft_1.ipynb`. It runs entirely in
the browser — the trained model gets exported to ONNX at build time, so
there's no backend to host or pay for. You can just push this to GitHub
Pages and it works.

## What's in this repo

```
index.html                    the app itself
assets/css/style.css          styling
assets/js/predict.js          feature engineering + runs the model
assets/js/ort/                onnxruntime-web, vendored locally
model/model.onnx              the trained model
model/scaler.json             scaler values + feature order, used to
                               preprocess inputs the same way training did
model/feature_importance.json feature importances, used for the "drivers"
                               list under each prediction
images/                        charts from the analysis, pre-rendered
notebook/MDS_Draft_1.ipynb    the original notebook
data/student_placement_data.csv
```

## Running it locally

```bash
python3 -m http.server 8000
```

then open `http://localhost:8000`. You can't just double-click
`index.html` and open it as a file — the browser blocks it from fetching
the model/JSON files when loaded that way. Needs an actual server, even a
local one.

## Deploying

```bash
git init
git add .
git commit -m "placement board"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

Then in the repo, go to Settings → Pages, set the source to the `main`
branch and root folder, save, and it'll be live in a minute or two at
`https://<username>.github.io/<repo>/`.

There's a `.nojekyll` file in here already — without it GitHub Pages runs
things through Jekyll by default, which can mess with folders like
`assets/js/ort/` that start with underscores or just behave oddly with a
lot of JS files. Keep it.

## How the model works, briefly

The notebook trains on 1,000 student records — CGPA, study hours, DSA
practice, projects, internships, interview prep, that kind of thing — and
adds a few engineered features on top:

- `academic_efficiency` = CGPA × study hours per day
- `practical_experience` = projects + 2 × internships
- `dsa_overall_score` = (DSA hours per week × problems solved) / 100,
  which replaces two raw columns that were basically measuring the same
  thing

Placement outcomes in the data are imbalanced (more placed than not), so
training uses SMOTE to oversample the minority class before fitting a
tuned XGBoost classifier.

For the site, the fitted model gets converted to ONNX with `onnxmltools`
and loaded in-browser with `onnxruntime-web`. I checked the ONNX output
against the original scikit-learn/XGBoost predictions and they match to
about 7 decimal places, so nothing gets lost in the conversion.

## If you retrain the model

Swap out the CSV, retrain using the same pipeline as the notebook, then
re-export:

```python
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

onnx_model = convert_xgboost(final_model, initial_types=[('float_input', FloatTensorType([None, 11]))])
with open('model/model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

You'll also need to update `model/scaler.json` (new `mean_` / `scale_`
from the scaler) and `model/feature_importance.json` (new
`feature_importances_`). The `feature_order` list in `scaler.json` has to
match whatever column order the model was actually trained on, or
predictions will be silently wrong.

## Notes to self

- The "drivers" shown under each prediction aren't a real explainability
  method (no SHAP or anything) — it's importance × how far the input is
  from the training average, signed by whether that feature correlates
  positively or negatively with placement. Good enough to be useful,
  not rigorous enough to lean on too hard.
- Sliders are bounded to the min/max seen in the training data, so the
  model isn't asked to extrapolate too far outside what it's seen.
