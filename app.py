#importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

#CACHED PREPROCESSING FUNCTION
@st.cache_data
def load_and_preprocess():
    #importing dataset
    df=pd.read_csv('dataset/dataset.csv')
    #dropping null values
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    #feature columns
    features=[ 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo']
    #grouping same songs on seperate rows with different genre names as one
    df = df.groupby(['track_name', 'artists'], as_index=False).agg({
    'danceability': 'first',
    'energy': 'first',
    'loudness': 'first',
    'speechiness': 'first',
    'acousticness': 'first',
    'instrumentalness': 'first',
    'liveness': 'first',
    'valence': 'first',
    'tempo': 'first',
    'explicit': 'first',
    'popularity': 'first',
    'track_genre': lambda x: list(set(x))
    })
    #creating a dataframe copy to prevent changes to original dataframe
    X_raw=df[features].copy()
    
    #standardising data using scikit learn
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X_raw).astype(np.float32)

    #adding weights to the features
    weights={
        'danceability':1, #danceability matters stronly
        'energy':1, #energy matters strongly
        'loudness':0.5, #loudness in dB of a song is less significant in recommendation
        'speechiness':0.55, #prevents clustering of rap music or speechy tracks together
        'acousticness':1, #acousticness matters strongly
        'instrumentalness':0.7, #prevents strong clustering of instrumental tracks
        'liveness':0.5, #reduces emphasis on whether the track is a live recording or not
        'valence':1, #mood of the song matters strongly
        'tempo':0.7 #tempo is not as important for a song recommendation
    }
    weighted_array=np.array([weights[col] for col in features])
    X_weighted=X_scaled*weighted_array

    #manually normalising the weighted matrix
    norms=np.linalg.norm(X_weighted,axis=1,keepdims=True)
    norms[norms==0]=1
    X_normalised=X_weighted/norms
    #---SONG SEARCH DICTIONARY---
    name_in_indices={}
    for i,name in enumerate(df['track_name']):
        key=name.lower()
        if key in name_in_indices:
            name_in_indices[key].append(i)
        else:
            name_in_indices[key]=[i]
    
    return df,X_normalised,name_in_indices
#run once (cached)
df,X_normalised,name_in_indices=load_and_preprocess()

#---RECOMMENDER ALGORITHM---
def recommend(song_index,top_n=10,allow_explicit=True):
    song_vector=X_normalised[song_index]
    #calculating cosine simularities of the song against other songs
    similarities=X_normalised @ song_vector
    #removing the chosen song from its own recommendations
    similarities[song_index]=-1
    if not allow_explicit:
        explicit_mask=df['explicit']==True
        similarities[explicit_mask]=-1
    
    # Soft genre boost
    song_genre = df.iloc[song_index]['track_genre']
    song_genres = df.iloc[song_index]['track_genre']

    genre_mask = df['track_genre'].apply(
        lambda genres: any(g in genres for g in song_genres)).values
    similarities[genre_mask] *= 1.1
    #adding popularity weight
    song_pop = df.iloc[song_index]['popularity']
    pop_values = df['popularity'].values
    pop_diff = np.abs(pop_values - song_pop)
    # convert difference into weight (smaller diff → bigger weight)
    pop_weight = 1 / (1 + pop_diff / 100)

    similarities *= pop_weight
    #sorting top n recommendations
    top_indices=np.argpartition(similarities,-top_n)[-top_n:]
    top_indices=top_indices[np.argsort(similarities[top_indices])[::-1]]

    return top_indices,similarities

#---STREAMLIT---
st.title("🎵 Music Recommender")
st.markdown("Made by Omkar Dey")

#initializes session state variables
if 'indices' not in st.session_state:
    st.session_state.indices=None
if 'selected_index' not in st.session_state:
    st.session_state.selected_index=None

#takes user song input
text_input=st.text_input("Enter exact song name")
#explicit song optional filter
allow_explicit=st.checkbox("Allow Explicit songs", value=True)

#searches songs of entered name
if st.button("Search"):
    key=text_input.strip().lower()
    if text_input.strip()=="":
        st.warning("Please enter a song name")
        st.session_state.indices=None
    if key not in name_in_indices:
        st.error("Song not found")
        st.session_state.indices=None
    else:
        st.session_state.indices=name_in_indices[key]
        st.session_state.selected_index=None
        #lets the user select the song they want recommendations for
if st.session_state.indices is not None:
    indices=st.session_state.indices
    options=["-- Select a song --"]+list(range(len(indices)))
    selected_option=st.selectbox(
        "Select correct song",
        options,
        format_func=lambda x:
            x if isinstance(x,str)
            else f"{df.iloc[indices[x]]['track_name']} | {df.iloc[indices[x]]['artists']}"
    )
    if selected_option != "-- Select a song --":
        st.session_state.selected_index=indices[selected_option]
    else:
        st.session_state.selected_index=None

    #shows recommended songs
    if st.session_state.selected_index is not None:
        results,similarities=recommend(st.session_state.selected_index,allow_explicit=allow_explicit)
        st.subheader(f"If you like {df.iloc[st.session_state.selected_index]['track_name']} " 
                     f"by {df.iloc[st.session_state.selected_index]['artists']}, you will also like :")
        for idx in results:
            score=90.9*similarities[idx]
            st.markdown(f"**{df.iloc[idx]['track_name']}** by *{df.iloc[idx]['artists']}* | Genre: {df.iloc[idx]['track_genre']}\t"
                        f"| `{score:.1f}`% similarity")
