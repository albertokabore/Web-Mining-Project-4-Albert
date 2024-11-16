# Requests, JSON, and basic NLP with spaCy

Complete the tasks in the Python Notebook in this repository.
To be submitted for credit, all changes must be committed and pushed to this repository (do not create your own repository unless instructed to on the course website).

## Rubric

* (Question 1) Lyrics printed: 1 pt
* (Question 1) File created and submitted with notebook: 1 pt
* (Question 2) Correct polarity reported: 1 pt
* (Question 2) Question answered thoughtfully: 1 pt
* (Question 3) Function defined as specified: 1 pt
* (Question 3) Song lyrics retrieved and stored in separate files (0.5 pts/song): 2 pts
* (Question 4) Polarity scores printed (with appropriate label containing song title, .25 pts/song): 1 pt
* (Question 4) Questions answered thoughtfully: 2 pts

# Web Mining and Applied NLP (44-620)

## Requests, JSON, and NLP

### Student Name: Albert Kabore

Link to GitHub repo: https://github.com/albertokabore/Web-Mining-Project-4-Albert

Perform the tasks described in the Markdown cells below.  When you have completed the assignment make sure your code cells have all been run (and have output beneath them) and ensure you have committed and pushed ALL of your changes to your assignment repository.

Make sure you have [installed spaCy and its pipeline](https://spacy.io/usage#quickstart) and [spaCyTextBlob](https://spacy.io/universe/project/spacy-textblob)

Every question that requires you to write code will have a code cell underneath it; you may either write your entire solution in that cell or write it in a python file (`.py`), then import and run the appropriate code to answer the question.

This assignment requires that you write additional files (either JSON or pickle files); make sure to submit those files in your repository as well.

```python
import requests
import json

result = json.loads(requests.get('https://api.lyrics.ovh/v1/They Might Be Giants/Birdhouse in your soul').text)
```

```python

import json
import pickle

import requests
import spacy
from spacy.tokens import Doc
from spacytextblob.spacytextblob import SpacyTextBlob

print('All prereqs installed.')
!pip list
```
```python

# Step 1: Fetch lyrics from the API
url = 'https://api.lyrics.ovh/v1/They Might Be Giants/Birdhouse in your soul'
response = requests.get(url)
result = json.loads(response.text)

# Step 2: Save the result to a JSON file
filename = 'lyrics.json'
with open(filename, 'w') as json_file:
    json.dump(result, json_file)

print(f"Lyrics saved to {filename}.")
```

2. Read in the contents of your file.  Print the lyrics of the song (not the entire dictionary!) and use spaCyTextBlob to perform sentiment analysis on the lyrics.  Print the polarity score of the sentiment analysis.  Given that the range of the polarity score is `[-1.0,1.0]` which corresponds to how positive or negative the text in question is, do you think the lyrics have a more positive or negative connotaion?  Answer this question in a comment in your code cell.

```python

# Step 1: Read the contents of the file
with open('lyrics.json', 'r') as json_file:
    data = json.load(json_file)

# Step 2: Extract and print the lyrics (not the entire dictionary)
lyrics = data.get("lyrics", "")  # Extract only the lyrics field
print("Lyrics of the song:\n", lyrics)

# Step 3: Perform sentiment analysis using spaCyTextBlob
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Step 4: Add spacytextblob to the pipeline
if 'spacytextblob' not in nlp.pipe_names:
    nlp.add_pipe('spacytextblob')

# Explicitly call set_extension if needed (this step is optional with spaCyTextBlob, but shown for clarity)
Doc.set_extension('polarity', getter=lambda doc: doc._.blob.polarity, force=True)

# Analyze the lyrics
doc = nlp(lyrics)
polarity = doc._.polarity  # Access polarity using the custom extension

# Step 5: Print the polarity score
print(f"\nPolarity score: {polarity}")

# Step 6: Analyze and interpret the polarity score
# Comment: The polarity score ranges from [-1.0, 1.0]. Positive values indicate positive sentiment, 
# while negative values indicate negative sentiment. Zero indicates neutrality.
if polarity > 0:
    print("The lyrics have a more positive connotation.")
elif polarity < 0:
    print("The lyrics have a more negative connotation.")
else:
    print("The lyrics have a neutral connotation.")
```

3. Write a function that takes an artist, song, and filename, accesses the lyrics.ovh api to get the song lyrics, and writes the results to the specified filename.  Test this function by getting the lyrics to any four songs of your choice and storing them in different files.

```python
import requests
import json

def fetch_and_save_lyrics(artist, song, filename):
    """
    Fetches song lyrics from the lyrics.ovh API and saves the result to a file.
    
    Parameters:
    artist (str): Name of the artist.
    song (str): Name of the song.
    filename (str): Filename to save the lyrics.
    """
    url = f"https://api.lyrics.ovh/v1/{artist}/{song}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        with open(filename, 'w') as file:
            json.dump(data, file)
        print(f"Lyrics for '{song}' by '{artist}' saved to {filename}.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching lyrics for '{song}' by '{artist}': {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for '{song}' by '{artist}'.")

# Test the function with four different songs
fetch_and_save_lyrics("Taylor Swift", "Love Story", "love_story_lyrics.json")
fetch_and_save_lyrics("Ed Sheeran", "Shape of You", "shape_of_you_lyrics.json")
fetch_and_save_lyrics("Adele", "Hello", "hello_lyrics.json")
fetch_and_save_lyrics("Coldplay", "Yellow", "yellow_lyrics.json")
```

4. Write a function that takes the name of a file that contains song lyrics, loads the file, performs sentiment analysis, and returns the polarity score.  Use this function to print the polarity scores (with the name of the song) of the three files you created in question 3.  Does the reported polarity match your understanding of the song's lyrics? Why or why not do you think that might be?  Answer the questions in either a comment in the code cell or a markdown cell under the code cell.

```python
def analyze_sentiment(filename):
    """
    Reads song lyrics from a file, performs sentiment analysis, and returns the polarity score.
    
    Parameters:
    filename (str): The name of the file containing song lyrics.
    
    Returns:
    float: Polarity score of the song lyrics.
    """
    try:
        # Load the lyrics from the file
        with open(filename, 'r') as file:
            data = json.load(file)
        
        # Extract lyrics
        lyrics = data.get("lyrics", "")
        if not lyrics:
            print(f"No lyrics found in {filename}.")
            return None
        
        # Perform sentiment analysis
        nlp = spacy.load("en_core_web_sm")
        if 'spacytextblob' not in nlp.pipe_names:
            nlp.add_pipe('spacytextblob')
        
        doc = nlp(lyrics)
        return doc._.polarity
    
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {filename}.")
        return None

# Analyze the sentiment of the three files
songs = [
    ("Taylor Swift", "Love Story", "love_story_lyrics.json"),
    ("Ed Sheeran", "Shape of You", "shape_of_you_lyrics.json"),
    ("Adele", "Hello", "hello_lyrics.json"),
    ("Coldplay", "Yellow", "yellow_lyrics.json")
]

```











