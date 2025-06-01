from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import nltk
from nltk.stem import PorterStemmer
from nltk.chunk import RegexpParser

s = '''Good muffins (b.j. naga ganesh) cost $3.88\nin New York.  Please buy me
... two of them.\n\nThanks.'''
word_tokenize(s)
print(word_tokenize(s) )
print(sent_tokenize(s))


nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(s)
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

#stemming
ps = PorterStemmer()
 
# choose some words to be stemmed
words = ["program", "programs", "programmer", "programming", "programmers"]
 
for w in words:
    print(w, " : ", ps.stem(w))

#parts of speech
word_tokens = word_tokenize(s)

tagged = nltk.pos_tag(word_tokens)
print(tagged)


from nltk.chunk import RegexpParser

#sentence="Educative Answers is a free web encyclopedia written by devs for devs."
sentence="BMS College for Women in New York is the best college to study BCA"

tokens= word_tokenize(sentence)

#POS Tagging
pos_tags=nltk.pos_tag(tokens)

chunk_patterns = r"""
    NP: {<DT>?<JJ>*<NN>}  # Chunk noun phrases
    VP: {<VB.*><NP|PP>}  # Chunk verb phrases
"""

#chunking
# Create a chunk parser
chunk_parser = RegexpParser(chunk_patterns)

# Perform chunking
result = chunk_parser.parse(pos_tags)

# Print the chunked result
print(result)
result.draw()


namedEnt = nltk.ne_chunk(pos_tags, binary=False)
namedEnt.draw()