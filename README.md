# Vibrance

A content-based music recommendation system that suggests similar songs based on audio features, built with Python and Streamlit.

---

## Overview

Enter a song name and get 10 recommendations based on audio similarity. The system uses weighted cosine similarity across 9 audio features, with optional explicit content filtering and popularity-aware ranking.

## Features

* **Content-Based Filtering** — Recommends songs using weighted cosine similarity across audio features
* **Custom Feature Weighting** — Uses manually selected weights for danceability, energy, valence, acousticness, and other features
* **Explicit Filter** — Option to exclude explicit tracks
* **Genre Boosting** — Gives a small boost to songs sharing a genre with the selected track
* **Popularity Matching** — Soft-weights recommendations toward songs with similar popularity scores
* **Duplicate Handling** — Groups songs appearing across multiple genres into a single entry
* **Cached Preprocessing** — Uses `@st.cache_data` to avoid repeating preprocessing on every interaction
* **Streamlit UI** — Interactive song search, song disambiguation, and recommendations
* **Deployed App** — Available as a live Streamlit application

## Tech Stack

| Library        | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| `scikit-learn` | Feature standardization using `StandardScaler`         |
| `NumPy`        | Feature normalization and cosine similarity            |
| `Pandas`       | Dataset loading, cleaning, deduplication, and grouping |
| `Streamlit`    | Interactive web interface                              |

## How It Works

1. **Preprocessing** — The dataset is cleaned and songs appearing across multiple genre entries are grouped into single entries.
2. **Scaling** — 9 audio features are standardized using `StandardScaler`.
3. **Weighting** — Each feature is multiplied by a manually selected weight to control its contribution to similarity.
4. **Normalization** — The weighted feature vectors are L2-normalized.
5. **Recommendation** — The selected song is compared with all other songs using cosine similarity. Genre boosting and popularity weighting are then applied before returning the top 10 recommendations.

### Audio Features Used

| Feature          | Weight |
| ---------------- | -----: |
| Danceability     |    1.0 |
| Energy           |    1.0 |
| Valence          |    1.0 |
| Acousticness     |    1.0 |
| Instrumentalness |    0.7 |
| Tempo            |    0.7 |
| Speechiness      |   0.55 |
| Loudness         |    0.5 |
| Liveness         |    0.5 |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/music-recommender.git
cd music-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the dataset CSV at:

```text
dataset/dataset.csv
```

The CSV should contain:

```text
track_name
artists
explicit
popularity
track_genre
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
```

### 4. Run the app

```bash
streamlit run app.py
```

## Usage

1. Enter an exact song name.
2. If multiple songs have the same name, select the correct artist from the dropdown.
3. Toggle the explicit-content filter if required.
4. View the 10 recommended tracks and their recommendation scores.

## Project Structure

```text
music-recommender/
│
├── app.py
├── requirements.txt
├── dataset/
│   └── dataset.csv
└── README.md
```

## Notes

* Song search is an **exact match** and is case-insensitive.
* The recommender is **content-based** and does not use listening history or collaborative filtering.
* The recommendation score combines audio similarity with genre and popularity adjustments.

## Live Demo

[Music Recommender](https://music-recommender-by-omkar.streamlit.app/)

## Author

Made by **Omkar Dey**

## License

MIT License
