import nltk
from nltk.corpus import stopwords
import os
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
import numpy as np
from PIL import Image

nltk.download('wordnet')
nltk.download('omw-1.4')

content = []
for filename in os.listdir('holmes/'):
    with open(f'holmes/{filename}') as f:
        content.append(f.read())

corpus = []
for item in content:
    corpus.extend([word.lower() for word in nltk.word_tokenize(item)])

corpus = [w for w in corpus if w not in stopwords.words('english')]

corpus = [w for w in corpus if w.isalnum()]


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


corpus = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in corpus]


unique_string = " ".join(corpus)

wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
wordcloud.to_file("word_cloud.png")

unique_string_v2 = " ".join(corpus)
cloud_mask = np.array(Image.open("cloud.png"))
wordcloud = WordCloud(width=1000, height=500, background_color="white",
                      mask=cloud_mask, max_words=5000, contour_width=2, contour_color='black')
wordcloud.generate(unique_string_v2)
wordcloud.to_file("word_cloud_masked.png")