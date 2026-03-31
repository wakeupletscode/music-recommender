# 🎵 Music Recommender

A content-based music recommendation system that suggests similar songs based on audio features, built with Python and Streamlit.

---

## 📌 Overview

Enter any song name and get 10 personalized recommendations based on audio similarity. The system uses weighted cosine similarity across 9 audio features — with optional explicit content filtering — to surface tracks that genuinely match the feel of your chosen song.

---

## ✨ Features

- 🎯 **Content-Based Filtering** — Recommends songs using weighted cosine similarity across audio features
- ⚖️ **Custom Feature Weighting** — Manually tuned weights for danceability, energy, valence, acousticness, and more
- 🔇 **Explicit Filter** — Optional toggle to exclude explicit tracks from recommendations
- 📈 **Popularity Matching** — Soft-weights recommendations toward songs with similar popularity scores
- 🔎 **Duplicate Handling** — Groups songs appearing across multiple genres into single unified entries
- ⚡ **Cached Preprocessing** — Uses `@st.cache_data` so the dataset loads and scales only once
- 🌐 **Streamlit UI** — Clean, interactive web interface with song search and disambiguation

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | Feature standardization (StandardScaler) |
| `NumPy` | Cosine similarity via matrix dot product, normalization |
| `Pandas` | Dataset loading, deduplication, grouping |
| `Streamlit` | Interactive web UI |

---

## 🧠 How It Works

1. **Preprocessing** — The dataset is cleaned, deduplicated, and grouped so songs appearing in multiple genres are merged into one entry
2. **Scaling** — 9 audio features are standardized using `StandardScaler`
3. **Weighting** — Each feature is multiplied by a manually tuned weight to reflect its importance in music similarity
4. **Normalization** — The weighted feature matrix is L2-normalized row-wise for accurate cosine similarity
5. **Recommendation** — On query, the system computes dot product similarity between the selected song vector and all others, applies popularity weighting, and returns the top 10 matches

### Audio Features Used

| Feature | Weight | Reason |
|---|---|---|
| Danceability | 1.0 | Strong indicator of song feel |
| Energy | 1.0 | Core similarity signal |
| Valence | 1.0 | Mood of the song |
| Acousticness | 1.0 | Acoustic vs electronic character |
| Instrumentalness | 0.7 | Reduced to avoid over-clustering instrumentals |
| Tempo | 0.7 | Less critical than mood/energy |
| Speechiness | 0.55 | Reduced to prevent rap clustering |
| Loudness | 0.5 | Less perceptually meaningful |
| Liveness | 0.5 | Minimizes live recording bias |

---

## ⚙️ Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/music-recommender.git
cd music-recommender
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Place your dataset CSV at:
```
dataset/dataset.csv
```

The CSV should contain columns: `track_name`, `artists`, `explicit`, `popularity`, `track_genre`, and the 9 audio feature columns listed above.

**4. Run the app**
```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Type an exact song name into the search box
2. Select the correct version from the disambiguation dropdown (handles duplicate song names across artists)
3. Toggle explicit filter if needed
4. View your 10 recommendations with similarity scores

---

## 📂 Project Structure

```
music-recommender/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── dataset/
│   └── dataset.csv         # Spotify audio features dataset
└── README.md
```

---

## 📝 Notes

- Song search is **exact match** (case-insensitive) — ensure the song name is spelled correctly
- The recommender is purely **content-based** — it does not use listening history or collaborative filtering
- Similarity scores are displayed as a normalized percentage for readability

---

## 👤 Author

Made by **Omkar Dey**

---

## 📄 License

MIT License
