import os
import joblib
import re
import string
import numpy as np
import nltk
from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

nltk.data.path.append("../nltk_data")

class TweetCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        self.userPattern = '@[^\s]+'
        self.some = 'amp,today,tomorrow,going,girl'
        self.stopword = set(stopwords.words('english'))
        self.abbreviations = {
            "$" : " dollar ", "€" : " euro ", "4ao" : "for adults only", "a.m" : "before midday",
            "a3" : "anytime anywhere anyplace", "aamof" : "as a matter of fact", "acct" : "account",
            "adih" : "another day in hell", "afaic" : "as far as i am concerned", "afaict" : "as far as i can tell",
            "afaik" : "as far as i know", "afair" : "as far as i remember", "afk" : "away from keyboard",
            "app" : "application", "approx" : "approximately", "apps" : "applications",
            "asap" : "as soon as possible", "asl" : "age, sex, location", "atk" : "at the keyboard",
            "ave." : "avenue", "aymm" : "are you my mother", "ayor" : "at your own risk", 
            "b&b" : "bed and breakfast", "b+b" : "bed and breakfast", "b.c" : "before christ",
            "b2b" : "business to business", "b2c" : "business to customer", "b4" : "before",
            "b4n" : "bye for now", "b@u" : "back at you", "bae" : "before anyone else",
            "bak" : "back at keyboard", "bbbg" : "bye bye be good", "bbc" : "british broadcasting corporation",
            "bbias" : "be back in a second", "bbl" : "be back later", "bbs" : "be back soon",
            "be4" : "before", "bfn" : "bye for now", "blvd" : "boulevard", "bout" : "about",
            "brb" : "be right back", "bros" : "brothers", "brt" : "be right there",
            "bsaaw" : "big smile and a wink", "btw" : "by the way", "bwl" : "bursting with laughter",
            "c/o" : "care of", "cet" : "central european time", "cf" : "compare",
            "cia" : "central intelligence agency", "csl" : "can not stop laughing", "cu" : "see you",
            "cul8r" : "see you later", "cv" : "curriculum vitae", "cwot" : "complete waste of time",
            "cya" : "see you", "cyt" : "see you tomorrow", "dae" : "does anyone else",
            "dbmib" : "do not bother me i am busy", "diy" : "do it yourself", "dm" : "direct message",
            "dwh" : "during work hours", "e123" : "easy as one two three", "eet" : "eastern european time",
            "eg" : "example", "embm" : "early morning business meeting", "encl" : "enclosed",
            "encl." : "enclosed", "etc" : "and so on", "faq" : "frequently asked questions",
            "fawc" : "for anyone who cares", "fb" : "facebook", "fc" : "fingers crossed",
            "fig" : "figure", "fimh" : "forever in my heart", "ft." : "feet", "ft" : "featuring",
            "ftl" : "for the loss", "ftw" : "for the win", "fwiw" : "for what it is worth",
            "fyi" : "for your information", "g9" : "genius", "gahoy" : "get a hold of yourself",
            "gal" : "get a life", "gcse" : "general certificate of secondary education",
            "gfn" : "gone for now", "gg" : "good game", "gl" : "good luck", "glhf" : "good luck have fun",
            "gmt" : "greenwich mean time", "gmta" : "great minds think alike", "gn" : "good night",
            "g.o.a.t" : "greatest of all time", "goat" : "greatest of all time", "goi" : "get over it",
            "gps" : "global positioning system", "gr8" : "great", "gratz" : "congratulations",
            "gyal" : "girl", "h&c" : "hot and cold", "hp" : "horsepower", "hr" : "hour",
            "hrh" : "his royal highness", "ht" : "height", "ibrb" : "i will be right back",
            "ic" : "i see", "icq" : "i seek you", "icymi" : "in case you missed it", "idc" : "i do not care",
            "idgadf" : "i do not give a damn fuck", "idgaf" : "i do not give a fuck", "idk" : "i do not know",
            "ie" : "that is", "i.e" : "that is", "ifyp" : "i feel your pain", "IG" : "instagram",
            "iirc" : "if i remember correctly", "ilu" : "i love you", "ily" : "i love you",
            "imho" : "in my humble opinion", "imo" : "in my opinion", "imu" : "i miss you",
            "iow" : "in other words", "irl" : "in real life", "j4f" : "just for fun", "jic" : "just in case",
            "jk" : "just kidding", "jsyk" : "just so you know", "l8r" : "later", "lb" : "pound",
            "lbs" : "pounds", "ldr" : "long distance relationship", "lmao" : "laugh my ass off",
            "lmfao" : "laugh my fucking ass off", "lol" : "laughing out loud", "ltd" : "limited",
            "ltns" : "long time no see", "m8" : "mate", "mf" : "motherfucker", "mfs" : "motherfuckers",
            "mfw" : "my face when", "mofo" : "motherfucker", "mph" : "miles per hour", "mr" : "mister",
            "mrw" : "my reaction when", "ms" : "miss", "mte" : "my thoughts exactly", "nagi" : "not a good idea",
            "nbc" : "national broadcasting company", "nbd" : "not big deal", "nfs" : "not for sale",
            "ngl" : "not going to lie", "nhs" : "national health service", "nrn" : "no reply necessary",
            "nsfl" : "not safe for life", "nsfw" : "not safe for work", "nth" : "nice to have",
            "nvr" : "never", "nyc" : "new york city", "oc" : "original content", "og" : "original",
            "ohp" : "overhead projector", "oic" : "oh i see", "omdb" : "over my dead body", "omg" : "oh my god",
            "omw" : "on my way", "p.a" : "per annum", "p.m" : "after midday", "pm" : "prime minister",
            "poc" : "people of color", "pov" : "point of view", "pp" : "pages", "ppl" : "people",
            "prw" : "parents are watching", "ps" : "postscript", "pt" : "point", "ptb" : "please text back",
            "pto" : "please turn over", "qpsa" : "what happens", "ratchet" : "rude",
            "rbtl" : "read between the lines", "rlrt" : "real life retweet", 
            "rofl" : "rolling on the floor laughing", "roflol" : "rolling on the floor laughing out loud",
            "rotflmao" : "rolling on the floor laughing my ass off", "rt" : "retweet", "ruok" : "are you ok",
            "sfw" : "safe for work", "sk8" : "skate", "smh" : "shake my head", "sq" : "square",
            "srsly" : "seriously", "ssdd" : "same stuff different day", "tbh" : "to be honest",
            "tbs" : "tablespooful", "tbsp" : "tablespooful", "tfw" : "that feeling when", "thks" : "thank you",
            "tho" : "though", "thx" : "thank you", "tia" : "thanks in advance", "til" : "today i learned",
            "tl;dr" : "too long i did not read", "tldr" : "too long i did not read", "tmb" : "tweet me back",
            "tntl" : "trying not to laugh", "ttyl" : "talk to you later", "u" : "you", "u2" : "you too",
            "u4e" : "yours for ever", "utc" : "coordinated universal time", "w/" : "with", "w/o" : "without",
            "w8" : "wait", "wassup" : "what is up", "wb" : "welcome back", "wtf" : "what the fuck",
            "wtg" : "way to go", "wtpa" : "where the party at", "wuf" : "where are you from",
            "wuzup" : "what is up", "wywh" : "wish you were here", "yd" : "yard",
            "ygtr" : "you got that right", "ynk" : "you never know", "zzz" : "sleeping bored and tired"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_X = []
        for tweet in X:
            processed = self._process_tweets(tweet)
            abbrev_processed = self._convert_abbrev_in_text(processed)
            final_text = " ".join([w for w in abbrev_processed.split() if len(w) > 3])
            cleaned_X.append(final_text)
        return np.array(cleaned_X)

    def _process_tweets(self, tweet):
        tweet = str(tweet)

        replacements = {
            r"he's": "he is", r"there's": "there is", r"We're": "We are", r"That's": "That is", 
            r"won't": "will not", r"they're": "they are", r"Can't": "Cannot", r"wasn't": "was not", 
            r"don\x89Ûªt": "do not", r"aren't": "are not", r"isn't": "is not", r"What's": "What is", 
            r"haven't": "have not", r"hasn't": "has not", r"There's": "There is", r"He's": "He is", 
            r"It's": "It is", r"You're": "You are", r"I'M": "I am", r"shouldn't": "should not", 
            r"wouldn't": "would not", r"i'm": "I am", r"I\x89Ûªm": "I am", r"I'm": "I am", 
            r"Isn't": "is not", r"Here's": "Here is", r"you've": "you have", r"you\x89Ûªve": "you have", 
            r"we're": "we are", r"what's": "what is", r"couldn't": "could not", r"we've": "we have", 
            r"it\x89Ûªs": "it is", r"doesn\x89Ûªt": "does not", r"It\x89Ûªs": "It is", 
            r"Here\x89Ûªs": "Here is", r"who's": "who is", r"I\x89Ûªve": "I have", r"y'all": "you all", 
            r"can\x89Ûªt": "cannot", r"would've": "would have", r"it'll": "it will", r"we'll": "we will", 
            r"wouldn\x89Ûªt": "would not", r"We've": "We have", r"he'll": "he will", r"Y'all": "You all", 
            r"Weren't": "Were not", r"Didn't": "Did not", r"they'll": "they will", r"they'd": "they would", 
            r"DON'T": "DO NOT", r"That\x89Ûªs": "That is", r"they've": "they have", r"i'd": "I would", 
            r"should've": "should have", r"You\x89Ûªre": "You are", r"where's": "where is", 
            r"Don\x89Ûªt": "Do not", r"we'd": "we would", r"i'll": "I will", r"weren't": "were not", 
            r"They're": "They are", r"Can\x89Ûªt": "Cannot", r"you\x89Ûªll": "you will", 
            r"I\x89Ûªd": "I would", r"let's": "let us", r"it's": "it is", r"can't": "cannot", 
            r"don't": "do not", r"you're": "you are", r"i've": "I have", r"that's": "that is", 
            r"doesn't": "does not", r"didn't": "did not", r"ain't": "am not", r"you'll": "you will", 
            r"I've": "I have", r"Don't": "do not", r"I'll": "I will", r"I'd": "I would", 
            r"Let's": "Let us", r"you'd": "You would", r"Ain't": "am not", r"Haven't": "Have not", 
            r"Could've": "Could have", r"youve": "you have", r"donå«t": "do not", r"yrs": "years", 
            r"hrs": "hours", r"2morow|2moro": "tomorrow", r"2day": "today", r"4got|4gotten": "forget", 
            r"b-day|bday": "b-day", r"mother's": "mother", r"mom's": "mom", r"dad's": "dad", 
            r"hahah|hahaha|hahahaha": "haha", r"lmao|lolz|rofl": "lol", r"thanx|thnx": "thanks", 
            r"goood": "good", r"some1": "someone"
        }
        for pattern, replacement in replacements.items():
            tweet = re.sub(pattern, replacement, tweet)
            
        tweet = tweet.lower()
        tweet = tweet[1:]
        tweet = re.sub(self.urlPattern, '', tweet)
        tweet = re.sub(self.userPattern, '', tweet) 
        tweet = re.sub(self.some, '', tweet)
        tweet = tweet.translate(str.maketrans("","",string.punctuation))
        tokens = word_tokenize(tweet)
        final_tokens = [w for w in tokens if w not in self.stopword]
        wordLemm = WordNetLemmatizer()
        finalwords = []
        for w in final_tokens:
            if len(w) > 1:
                word = wordLemm.lemmatize(w)
                finalwords.append(word)
        return ' '.join(finalwords)

    def _convert_abbrev_in_text(self, tweet):
        words = tweet.split()
        t = [self.abbreviations[w.lower()] if w.lower() in self.abbreviations.keys() else w for w in words]
        return ' '.join(t)
app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), '../sentiment_pipeline.pkl')
pipeline = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        user_input = request.form.get('tweet_text')
        
        if user_input:
            pred_value = pipeline.predict([user_input])[0]
            if pred_value == 4:
                prediction = "Positive"
            else:
                prediction = "Negative"
            
    return render_template('index.html', prediction=prediction)

