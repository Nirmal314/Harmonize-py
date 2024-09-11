# Harmonize

This project categorizes songs from a Spotify playlist into different categories using machine learning techniques.

## Setup

1. Install the required Python packages:

   ```
   pip<version> install -r requirements.txt
   ```

2. Set up your Spotify Developer account and create an app to get your `client_id` and `client_secret`.

3. Create .env file and replace the `CLIENT_ID` and `CLIENT_SECRET` with your actual credentials.

4. Replace `your_playlist_id` in `song_categorization.py` with the ID of the Spotify playlist you want to analyze.

## Running the Project

Run the Python script to categorize songs:

```
python<version> song_categorization.py
```

The script will output the categorization results and cluster analysis.
