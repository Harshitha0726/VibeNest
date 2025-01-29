import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
songs = pd.read_csv('data.csv')

# Define the classify_song_emotion function
def classify_song_emotion(valence, energy):
    if valence > 0.7 and energy > 0.7:
        return 'happy'
    elif valence < 0.3 and energy < 0.4:
        return 'sad'
    elif valence > 0.5 and energy < 0.4:
        return 'calm'
    elif valence < 0.4 and energy > 0.7:
        return 'angry'
    else:
        return 'neutral'

# Add a new column 'emotion' to the dataset based on valence and energy
songs['emotion'] = songs.apply(lambda row: classify_song_emotion(row['valence'], row['energy']), axis=1)

# Map emotion categories to emojis
mood_map = {
    'happy': '😊',
    'sad': '😔',
    'calm': '😌',
    'angry': '😠',
    'neutral': '😐'
}
songs['mood'] = songs['emotion'].map(mood_map)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(songs['mood'], songs['emotion'], test_size=0.2, random_state=42)

# Convert emojis into numerical features using TF-IDF vectorizer
vectorizer = TfidfVectorizer(token_pattern=r'\S')  # Treat each emoji as a separate token
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Define a function to recommend music based on the input emoji
def recommend_music(emoji):
    # Check if the emoji is in the mood_map
    if emoji not in mood_map.values():
        print("Emoji not recognized. Please use one of the following: 😊, 😔, 😠, 😌, 😐")
        return
    
    # Convert the emoji into a numerical feature using the TF-IDF vectorizer
    emoji_tfidf = vectorizer.transform([emoji])
    
    # Predict the emotion category using the trained model
    emotion = model.predict(emoji_tfidf)[0]
    
    # Find songs matching the predicted emotion category
    matching_songs = songs[songs['emotion'] == emotion]
    
    if matching_songs.empty:
        print(f"No songs found for the emotion '{emotion}'.")
        return
    
    # Get the top 10 most popular songs with the matching emotion
    top_songs = matching_songs.sort_values(by='popularity', ascending=False).head(10)
    
    # Generate clickable hyperlinks for the songs
    print(f"\nTop songs for the emotion '{emotion}' ({emoji}):\n")
    for index, song in top_songs.iterrows():
        song_name = song['name']
        song_artists = song['artists']
        song_id = song['id']
        song_url = f"https://open.spotify.com/track/{song_id}"
        
        # Print the song name, artists, and clickable URL
        print(f"Song: {song_name}\nArtists: {song_artists}\nLink: {song_url}\n")

