import pandas as pd
import Capture_emotion 

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

def classify_emotion(user_input):
    emotion_categories = {
        'happy': ['happy', 'joyful', 'upbeat'],
        'sad': ['sad', 'depressed', 'melancholic'],
        'calm': ['calm', 'relaxed'],
        'angry': ['angry', 'frustrated', 'irritated'],
        'neutral': ['neutral', 'okay']
    }
    for category, keywords in emotion_categories.items():
        if any(keyword in user_input.lower().split() for keyword in keywords):
            return category
    return 'neutral'

def recommend_songs(emotion):
    # Load the dataset
    try:
        songs = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: The dataset 'data.csv' was not found.")
        return

    # Add a new column 'emotion' to the dataset based on valence and energy
    songs['emotion'] = songs.apply(lambda row: classify_song_emotion(row['valence'], row['energy']), axis=1)

    # Find songs matching the given emotion
    matching_songs = songs[songs['emotion'] == emotion]
    
    if matching_songs.empty:
        print("No songs found for this emotion.")
        return

    # Get the top 10 most popular songs with the matching emotion
    top_songs = matching_songs.sort_values(by='popularity', ascending=False).head(10)
    
    # Generate clickable hyperlinks for the songs
    for index, song in top_songs.iterrows():
        song_name = song['name']
        song_artists = song['artists']
        song_id = song['id']
        song_url = f"https://open.spotify.com/track/{song_id}"
        
        # Print the song name, artists, and clickable URL
        print(f"Song: {song_name}\nArtists: {song_artists}\nLink: {song_url}\n")

def get_user_emotion_by_text():
    user_input = input("Enter your current emotion (e.g. happy, sad, calm, angry, neutral): ")
    return classify_emotion(user_input)

def get_user_emotion_by_camera():
    emotion_label = Capture_emotion.detect_emotion()  # Call the emotion detection function
    return emotion_label

def main():
    print("Choose an option to input your emotion:")
    print("1. Text input")
    print("2. Camera input")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        user_emotion = get_user_emotion_by_text()
    elif choice == '2':
        print("Please look at the camera...")
        user_emotion = get_user_emotion_by_camera()
    else:
        print("Invalid choice. Exiting.")
        return

    print(f"Detected Emotion: {user_emotion}")
    recommend_songs(user_emotion)

if __name__ == "__main__":
    main()
