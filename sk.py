import sklearn
from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
movie_train = load_files('aclImdb/train', shuffle=True)
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(movie_train.data)
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)
docs_train, docs_test, y_train, y_test = train_test_split(movie_tfidf, movie_train.target, test_size=0.20, random_state=12)
clf = MultinomialNB().fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
print(sklearn.metrics.accuracy_score(y_test, y_pred))
