# Notebooks

The notebooks and data behind the articles on [archibaldchain.github.io](https://archibaldchain.github.io/articles.html).
Each folder is self-contained: the notebook, the exact data it reads, and a note on where that data comes from.

| Article | Notebook | Run it |
|---|---|---|
| [What drives Ontario's electricity demand?](https://archibaldchain.github.io/articles/ontario-load-weather-eda.html) | [`ontario-load-weather-eda/ontario-load-weather-eda.ipynb`](ontario-load-weather-eda/ontario-load-weather-eda.ipynb) | [Open in Colab](https://colab.research.google.com/github/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/ontario-load-weather-eda/ontario-load-weather-eda.ipynb) |

## Running locally

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt                      # or: uv pip install -r requirements.txt
jupyter lab
```

Open a notebook from inside its own folder so the relative `data/` path resolves.
In Google Colab nothing needs to be downloaded: the notebook fetches its data file from this repository.

## Licence

Code in these notebooks is MIT-licensed, like the rest of this repository.
The data files keep their original sources' terms; see each folder's `data/README.md`.
