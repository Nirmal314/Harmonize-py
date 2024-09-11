# Harmonize

This project categorizes songs from a Spotify playlist into different categories using machine learning techniques.

## Setup

1. Install the required Python packages:

   ```
   pip<version> install -r requirements.txt
   ```

2. Set up your Spotify Developer account and create an app to get your `client_id`, `client_secret`, and `redirect_uri`.

3. Replace the `your_client_id`, `your_client_secret`, and `your_redirect_uri` in the `song_categorization.py` script with your actual credentials.

4. Replace `your_playlist_id` with the ID of the Spotify playlist you want to analyze.

## Running the Project

Run the Python script to categorize songs:

```
python<version> song_categorization.py
```

The script will output the categorization results and cluster analysis.
