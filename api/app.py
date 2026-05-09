from flask import Flask, request, jsonify, render_template
import re
import string
import nltk
import numpy as np
import joblib
import os

nltk.data.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../nltk_data')))
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS_SET = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
URL_PATTERN = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
USER_PATTERN = '@[^\s]+'
SOME_WORDS = r'\b(amp|today|tomorrow|going|girl)\b'
LABEL_MAP = {4: "Positive", 0: "Negative"}

ABBREVIATIONS = {
    "$": "dollar", "€": "euro", "4ao": "for adults only", "a.m": "before midday",
    "a3": "anytime anywhere anyplace", "aamof": "as a matter of fact", "acct": "account",
    "adih": "another day in hell", "afaic": "as far as i am concerned", "afaict": "as far as i can tell",
    "afaik": "as far as i know", "afair": "as far as i remember", "afk": "away from keyboard",
    "app": "application", "approx": "approximately", "apps": "applications",
    "asap": "as soon as possible", "asl": "age sex location", "atk": "at the keyboard",
    "ave.": "avenue", "aymm": "are you my mother", "ayor": "at your own risk",
    "b&b": "bed and breakfast", "b+b": "bed and breakfast", "b.c": "before christ",
    "b2b": "business to business", "b2c": "business to customer", "b4": "before",
    "b4n": "bye for now", "b@u": "back at you", "bae": "before anyone else",
    "bak": "back at keyboard", "bbbg": "bye bye be good", "bbc": "british broadcasting corporation",
    "bbias": "be back in a second", "bbl": "be back later", "bbs": "be back soon",
    "be4": "before", "bfn": "bye for now", "blvd": "boulevard", "bout": "about",
    "brb": "be right back", "bros": "brothers", "brt": "be right there",
    "bsaaw": "big smile and a wink", "btw": "by the way", "bwl": "bursting with laughter",
    "c/o": "care of", "cet": "central european time", "cf": "compare",
    "cia": "central intelligence agency", "csl": "can not stop laughing", "cu": "see you",
    "cul8r": "see you later", "cv": "curriculum vitae", "cwot": "complete waste of time",
    "cya": "see you", "cyt": "see you tomorrow", "dae": "does anyone else",
    "dbmib": "do not bother me i am busy", "diy": "do it yourself", "dm": "direct message",
    "dwh": "during work hours", "e123": "easy as one two three", "eet": "eastern european time",
    "eg": "example", "embm": "early morning business meeting", "encl": "enclosed",
    "encl.": "enclosed", "etc": "and so on", "faq": "frequently asked questions",
    "fawc": "for anyone who cares", "fb": "facebook", "fc": "fingers crossed",
    "fig": "figure", "fimh": "forever in my heart", "ft.": "feet", "ft": "featuring",
    "ftl": "for the loss", "ftw": "for the win", "fwiw": "for what it is worth",
    "fyi": "for your information", "g9": "genius", "gahoy": "get a hold of yourself",
    "gal": "get a life", "gcse": "general certificate of secondary education",
    "gfn": "gone for now", "gg": "good game", "gl": "good luck", "glhf": "good luck have fun",
    "gmt": "greenwich mean time", "gmta": "great minds think alike", "gn": "good night",
    "g.o.a.t": "greatest of all time", "goat": "greatest of all time", "goi": "get over it",
    "gps": "global positioning system", "gr8": "great", "gratz": "congratulations",
    "gyal": "girl", "h&c": "hot and cold", "hp": "horsepower", "hr": "hour",
    "hrh": "his royal highness", "ht": "height", "ibrb": "i will be right back",
    "ic": "i see", "icq": "i seek you", "icymi": "in case you missed it", "idc": "i do not care",
    "idgadf": "i do not give a damn fuck", "idgaf": "i do not give a fuck", "idk": "i do not know",
    "ie": "that is", "i.e": "that is", "ifyp": "i feel your pain", "ig": "instagram",
    "iirc": "if i remember correctly", "ilu": "i love you", "ily": "i love you",
    "imho": "in my humble opinion", "imo": "in my opinion", "imu": "i miss you",
    "iow": "in other words", "irl": "in real life", "j4f": "just for fun", "jic": "just in case",
    "jk": "just kidding", "jsyk": "just so you know", "l8r": "later", "lb": "pound",
    "lbs": "pounds", "ldr": "long distance relationship", "lmao": "laugh my ass off",
    "lmfao": "laugh my fucking ass off", "lol": "laughing out loud", "ltd": "limited",
    "ltns": "long time no see", "m8": "mate", "mf": "motherfucker", "mfs": "motherfuckers",
    "mfw": "my face when", "mofo": "motherfucker", "mph": "miles per hour", "mr": "mister",
    "mrw": "my reaction when", "ms": "miss", "mte": "my thoughts exactly", "nagi": "not a good idea",
    "nbc": "national broadcasting company", "nbd": "not big deal", "nfs": "not for sale",
    "ngl": "not going to lie", "nhs": "national health service", "nrn": "no reply necessary",
    "nsfl": "not safe for life", "nsfw": "not safe for work", "nth": "nice to have",
    "nvr": "never", "nyc": "new york city", "oc": "original content", "og": "original",
    "ohp": "overhead projector", "oic": "oh i see", "omdb": "over my dead body", "omg": "oh my god",
    "omw": "on my way", "p.a": "per annum", "p.m": "after midday", "pm": "prime minister",
    "poc": "people of color", "pov": "point of view", "pp": "pages", "ppl": "people",
    "prw": "parents are watching", "ps": "postscript", "pt": "point", "ptb": "please text back",
    "pto": "please turn over", "qpsa": "what happens", "ratchet": "rude",
    "rbtl": "read between the lines", "rlrt": "real life retweet",
    "rofl": "rolling on the floor laughing", "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my ass off", "rt": "retweet", "ruok": "are you ok",
    "sfw": "safe for work", "sk8": "skate", "smh": "shake my head", "sq": "square",
    "srsly": "seriously", "ssdd": "same stuff different day", "tbh": "to be honest",
    "tbs": "tablespooful", "tbsp": "tablespooful", "tfw": "that feeling when", "thks": "thank you",
    "tho": "though", "thx": "thank you", "tia": "thanks in advance", "til": "today i learned",
    "tl;dr": "too long i did not read", "tldr": "too long i did not read", "tmb": "tweet me back",
    "tntl": "trying not to laugh", "ttyl": "talk to you later", "u": "you", "u2": "you too",
    "u4e": "yours for ever", "utc": "coordinated universal time", "w/": "with", "w/o": "without",
    "w8": "wait", "wassup": "what is up", "wb": "welcome back", "wtf": "what the fuck",
    "wtg": "way to go", "wtpa": "where the party at", "wuf": "where are you from",
    "wuzup": "what is up", "wywh": "wish you were here", "yd": "yard",
    "ygtr": "you got that right", "ynk": "you never know", "zzz": "sleeping bored and tired"
}

REPLACEMENTS = {
    r"he's": "he is", r"there's": "there is", r"we're": "we are", r"that's": "that is",
    r"won't": "will not", r"they're": "they are", r"can't": "cannot", r"wasn't": "was not",
    r"don't": "do not", r"aren't": "are not", r"isn't": "is not", r"what's": "what is",
    r"haven't": "have not", r"hasn't": "has not", r"he's": "he is", r"it's": "it is",
    r"you're": "you are", r"i'm": "i am", r"shouldn't": "should not", r"wouldn't": "would not",
    r"here's": "here is", r"you've": "you have", r"we've": "we have", r"couldn't": "could not",
    r"who's": "who is", r"y'all": "you all", r"would've": "would have", r"it'll": "it will",
    r"we'll": "we will", r"he'll": "he will", r"weren't": "were not", r"didn't": "did not",
    r"they'll": "they will", r"they'd": "they would", r"they've": "they have", r"i'd": "i would",
    r"should've": "should have", r"where's": "where is", r"we'd": "we would", r"i'll": "i will",
    r"let's": "let us", r"doesn't": "does not", r"ain't": "am not", r"you'll": "you will",
    r"i've": "i have", r"you'd": "you would", r"could've": "could have", r"youve": "you have",
    r"yrs": "years", r"hrs": "hours", r"2morow|2moro": "tomorrow", r"2day": "today",
    r"4got|4gotten": "forget", r"b-day|bday": "birthday", r"hahah|hahaha|hahahaha": "haha",
    r"lmao|lolz|rofl": "lol", r"thanx|thnx": "thanks", r"goood": "good", r"some1": "someone"
}

def preprocess_tweet_for_model(raw_tweet):
    tweet = str(raw_tweet).lower()
    for pattern, replacement in REPLACEMENTS.items():
        tweet = re.sub(r'\b' + pattern + r'\b', replacement, tweet)
    tweet = re.sub(URL_PATTERN, '', tweet)
    tweet = re.sub(USER_PATTERN, '', tweet)
    tweet = re.sub(SOME_WORDS, '', tweet)
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(tweet)
    tokens = [w for w in tokens if w not in STOPWORDS_SET]
    lemmatized = [LEMMATIZER.lemmatize(w) for w in tokens if len(w) > 1]
    processed_str = ' '.join(lemmatized)
    words = processed_str.split()
    expanded = [ABBREVIATIONS.get(w.lower(), w) for w in words]
    final_text = ' '.join(w for w in expanded if len(w) > 3)
    return np.array([final_text])

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

model_path = os.path.join(os.path.dirname(__file__), '../sentiment_pipeline.pkl')
try:
    pipeline = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {model_path}")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    tweet_text = ''
    if request.method == 'POST':
        tweet_text = request.form.get('tweet_text', '').strip()
        if tweet_text and len(tweet_text) <= 500:
            processed = preprocess_tweet_for_model(tweet_text)
            pred_value = pipeline.predict(processed)[0]
            prediction = LABEL_MAP.get(pred_value, "Unknown")
    return render_template('index.html', prediction=prediction, tweet_text=tweet_text)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'tweet' not in data:
        return jsonify({'error': 'Missing tweet field'}), 400
    user_input = data['tweet'].strip()
    if not user_input or len(user_input) > 500:
        return jsonify({'error': 'Tweet must be between 1 and 500 characters'}), 400
    processed = preprocess_tweet_for_model(user_input)
    pred_value = pipeline.predict(processed)[0]
    return jsonify({'sentiment': LABEL_MAP.get(pred_value, "Unknown")})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)