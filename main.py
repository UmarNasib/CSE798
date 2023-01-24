# Importing required module

import numpy as np
from nltk.tokenize import word_tokenize

# importing required modules
from PyPDF2 import PdfReader

# import fitz

# creating a pdf reader object
reader = PdfReader('inputPdf.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[0]

# extracting text from page
text = page.extract_text()
print(text)


# Preprocessing the text data
sentences = []
word_set = []

for sent in text:
    x = [i.lower() for i in word_tokenize(sent) if i.isalpha()]
    sentences.append(x)
    for word in x:
        if word not in word_set:
            word_set.append(word)

# Set of vocab
word_set = set(word_set)

# Total documents in our corpus
total_documents = len(sentences)

# Creating an index for each word in our vocab.
index_dict = {}  # Dictionary to store index for each word
i = 0
for word in word_set:
    index_dict[word] = i
    i += 1

# Create a count dictionary
def count_dict(sentences):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count

word_count = count_dict(sentences)

#Term Frequency
def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N

# Inverse Document Frequency
def inverse_doc_freq(word):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_documents / word_occurance)


def tf_idf(sentence):
    tf_idf_vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = termfreq(sentence, word)
        idf = inverse_doc_freq(word)

        value = tf * idf
        tf_idf_vec[index_dict[word]] = value
    return tf_idf_vec


# TF-IDF Encoded text corpus
vectors = []
for sent in sentences:
    vec = tf_idf(sent)
    vectors.append(vec)

print(vectors[0])

