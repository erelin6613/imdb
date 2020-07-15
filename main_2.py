#!/usr/bin/env python3

"""

The program designated to preform basic natural language
processing with help of movie reviews one can scrape
as needed.

"""

import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
from gensim.models import Word2Vec
import gensim.downloader as api

nltk.download('stopwords', quiet=True)

# name you app however you like
name = 'NLP helper'


"""
Function list_reviews prints in a user firendly manner the list of available
revies and returns a tuple of lists (reviews and file paths)
"""
def list_reviews():

	c = 0
	reviews = []
	filenames = []
	for each in os.listdir('.'):
		if 'reviews' in each:
			for file in os.listdir(os.path.join(os.getcwd(), each)):
				print('{}) {}'.format(c, file))
				filenames.append(file)
				c += 1
				#reviews.append(os.path.join(each, file))
				with open(os.path.join(each, file), 'r') as f:
					reviews.append(f.read())

	return (reviews, filenames)

"""
Function tokenize_text takes as an argument text(string) and returns
token list; used casual_tokenize() to remove occurances of scewed,
misspelt word in case one decides to perform similarity scoring
"""
def tokenize_text(text):
	tokens = casual_tokenize(text)
	return [word for word in tokens if word.isalpha() or word.isdigit()]


"""
Function load_reviews loads new reviews not yet listed in the directory
reviews; the file 'crawled.txt' obtained using the open source crawler
by Valentyna Fihurska, available at 
https://github.com/erelin6613/crawler/blob/master/whole_parser_copy.py
"""
def load_reviews():
	c = 0
	with open(os.path.join(os.getcwd(), 'crawled.txt')) as links:
		for url in links:
			if c == 500:
				break
			if url.strip().endswith('reviews') == True:
				r = requests.get(url.strip())
				soup = BeautifulSoup(r.text, 'lxml')

				for each in soup.find_all(class_='text show-more__control'):
					if (soup.find('title').text+'.txt') in set(os.listdir('reviews')):
						print('File', soup.find('title').text+'.txt', 'already exists')
						break
					tokens = tokenize_text(each.get_text())
					if len(tokens) > 300:
						with open(os.path.join('reviews', soup.find('title').text+'.txt'), 'w') as file:
							file.write(each.get_text())
							print('loaded review:', soup.find('title').text+'.txt')
							c += 1
							break

"""
Function get_similarity() takes a word and returns a list
of similar word. WARNING! The function requires to load 
a word2vec model once, gensim`s 'glove-wiki-gigaword-100' 
set by default (128 MB). Called from find_syns_n_matches()
"""
def get_similarity(word):

	word_vectors = api.load("glove-wiki-gigaword-100")
	return word_vectors.most_similar(word, topn=10)

"""
Function find_syns_n_matches() takes word (entered by user)
and tokens from the file (chosed by user) and returns
set of synonyms, set of similar words, number of ocurrances
abd relative occurances of the word in tokens list
"""
def find_syns_n_matches(word=None, tokens=None):
	synonyms = set()
	similar_words = set()
	lmt = WordNetLemmatizer()
	lemma = lmt.lemmatize(word)
	for w in wn.synsets(lemma):
		for syn in w.lemmas():
			for token in tokens:
				if str(syn.name()) == token.lower():
					synonyms.add(syn.name())
	for item in get_similarity(word):
		if item[0] in set(tokens):
			similar_words.add(item[0])

	occurances = 0
	for token in tokens:
		if str(token.lower()) == str(word):
			occurances += 1
	relative_occurances = occurances / len(tokens)

	return [synonyms, similar_words, occurances, relative_occurances]

"""
Function determine_sentiment() takes text of the file and
returns a sentiment score;
cite from SentimentIntensityAnalyzer`s documentation
(https://github.com/cjhutto/vaderSentiment)
'...Over 9,000 token features were rated on a scale from 
"[–4] Extremely Negative" to "[4] Extremely Positive", with allowance for 
"[0] Neutral (or Neither, N/A)". We kept every lexical feature that had a 
non-zero mean rating, and whose standard deviation was less than 2.5 as 
determined by the aggregate of those ten independent raters. 
This left us with just over 7,500 lexical features with validated valence 
scores that indicated both the sentiment polarity (positive/negative), 
and the sentiment intensity on a scale from –4 to +4. 
For example, the word "okay" has a positive valence of 0.9, "good" is 1.9, 
and "great" is 3.1, whereas "horrible" is –2.5, the frowning emoticon 
:( is –2.2, and "sucks" and it's slang derivative "sux" are both –1.5....'
(put simplier, compound score is -4 indicating the most negative intensity, 
0 - neutral, 4 - the most positive)
"""
def determine_sentiment(text):
	sa = SentimentIntensityAnalyzer()
	score = sa.polarity_scores(text)
	return score

"""
Function count_utterances() takes a text as an argument, calculates some
text processing statistics as a dictionary: term frequency document frequency(count_vector), 
term frequency–inverse document frequency(tfidf_vec), all tokens in corpus (all_tokens, these
correspond to frequencies in the count_vector and tfidf_vec), total amount of tokens
(total_tokens), stopwords in the text(stopwords), sentiment analysis metrics(semtiment_score)
and writes a dictionary to the file
"""
def count_utterances(text):
	counts = {}
	tokens = tokenize_text(text)
	counts['total_tokens'] = len(tokens)
	vectorizer = CountVectorizer()
	tfidf_vectorizer = TfidfVectorizer()
	vector = vectorizer.fit_transform(text.split('\n'))
	tfidf_vector = tfidf_vectorizer.fit_transform(text.split('\n'))
	counts['all_tokens'] = vectorizer.get_feature_names()
	counts['count_vector'] = vector.toarray()
	counts['tfidf_vec'] = tfidf_vector.toarray()
	stopw = set()
	for w in tokens: 
		if w in stopwords.words('english'): 
			stopw.add(w) 
	counts['stopwords'] = stopw
	counts['sentiment_score'] = determine_sentiment(text)

	return counts


"""
The basic user interface to call functions and get needed processes 
to be taken care of
"""
def ui_setup():
	print('\nWelcome to {}!'.format(name))
	while True:

		inp = input('How can I help?\nEnter corresponding number to perform an action.\
				\n1) List existing reviews \n 2) Load additional 500 reviews \
				\n3) Print and write to a file text analysis \
				\n4) Find specific word and its synonyms in file \
				\n5) Print a sentiment scores\n 6) Exit')
		print('')

		if inp == '1':
			list_reviews()
		elif inp == '2':
			load_reviews()
		elif inp == '3':
			list_reviews()
			choice = input('Please choice which file you want to analyze and type its number')
			with open(os.path.join('reviews', list_reviews()[1][int(choice)]), 'r') as f:
				text = f.read()
			counts = count_utterances(text)
			print(counts)
			with open(os.path.join('reviews', 'analysis_'+list_reviews()[1][int(choice)]), 'w') as file:
				for key in counts.keys():
					file.write('{}: {}\n'.format(key, counts[key]))
				print('Results are stored as', os.path.join('reviews', 'analysis_'+list_reviews()[1][int(choice)]))
		elif inp == '4':
			list_reviews()
			word = input('Please specify the word to check')
			file = input('Please specify the number of the file where to check')
			syns = find_syns_n_matches(word, tokenize_text(list_reviews()[0][int(file)]))
			print('synonyms:', syns[0], '\nsimilar words in corpus','\noccurances:', syns[2], '\nrelative_occurances', syns[3])
		elif inp == '5':
			list_reviews()
			f_choice = input('Please choose the file to count the score for.')
			with open(os.path.join('reviews', list_reviews()[1][int(f_choice)]), 'r') as file:
				scores = determine_sentiment(file.read())
			print(scores)
		elif inp == '6':
			break
		else:
			print('Please enter a valid choice.')



if __name__ == '__main__':
	ui_setup()