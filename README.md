# Natural Language Processing
### what is natural language processing?
Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.

NLP combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. Together, these technologies enable computers to process human language in the form of text or voice data and to ‘understand’ its full meaning, complete with the speaker or writer’s intent and sentiment.

NLP drives computer programs that translate text from one language to another, respond to spoken commands, and summarize large volumes of text rapidly—even in real time. There’s a good chance you’ve interacted with NLP in the form of voice-operated GPS systems, digital assistants, speech-to-text dictation software, customer service chatbots, and other consumer conveniences. But NLP also plays a growing role in enterprise solutions that help streamline business operations, increase employee productivity, and simplify mission-critical business processes.

### Basic Text Preprocessing
#### Normalization
change all letters to uppercase or lowercase.
#### Tokenization
tokenization will break the entire text into words, or sentences.
#### Punctuation Removal
punctuation eraser
#### Alphanumeric Cleansing
just make sure that the variable consists of letters and numbers.
#### Stopwords Removal
are words that often / always appear in conversational language, stopwords is not absolute and depends on what the domain is talking about.
### Why Preprocessing
Vocabuary will be used as a feature. Example(in indonesian language):

- Ini adalah pensil
- Ini adalah pulpen
- Saya beli pensin ini    
- Saya beli pulpen itu
 
Tokens:
- Ini 
- adalah
- pensil
- pulpen
- saya
- beli
- itu
This means that only with these tokens we can make the text sentence above.
Then the token/vovabuary will be compiled into a table.

![image](https://user-images.githubusercontent.com/86812576/168838238-fb84353c-5747-4821-9e41-4a83ded2920a.png)

The left side is as a line (document). And the right part is the feature(vocab) taken from the token.
If there is the word "Hello" with "hello" is a different token even though the meaning is the same, unless it is normalized.

### What to do after vocab/token becomes a feature?
### Encode
### 1. Bag of Words (BoW)
![image](https://user-images.githubusercontent.com/86812576/169047358-53300fd6-42a0-4966-a565-fef1478fddd6.png)

The task of Bag of Words is to make the text into a vector by counting the tokens.
When the vocab/token has been counted with the encoder, all you have to do is look for the pattern with machine learning.

### Inverse Document Frequency
![image](https://user-images.githubusercontent.com/86812576/169047814-4f881a18-73c3-4232-b932-f155558d7df4.png)

The idea is to invert the DF value. Why?

For example, there is a token that appears in a particular document, meaning that it's a keyword for that document. Example, the word "itu" (the example above) has a value of 1/4 and only appears in certain documents and not in other documents, it is specific in certain documents so that it is weighted more strongly (From 1/4 to 4 because it is reversed). Another example is that the word “ini” appears in many documents, because it often appears in many documents, means that the word “ini” is not an important word because it does not distinguish between documents. Actually IDF is scaling for documents.

### 2. Term Frequency - Inverse Document Frequency (TFIDF)
TFIDF is BoW which is scaled using IDF, which words are important and which words often appear are not important. For example _Stopwords_ that almost appear in the entire document. Usually words that often appear in other documents will be weakened, because the goal is to have a more meaningful encoding. Meanwhile, specific words rarely appear to be strengthened, so that the TFIDF will be decimalized.
That's why there are 2 options, whether to remove stopwords, or use both.
Because the function of TFIDF is also to weaken _Stopwords_. This will be done in text preprocessing before entering the machine.

# Document Similarity

![Screenshot 2022-05-12 211659](https://user-images.githubusercontent.com/86812576/168096561-c048a189-2377-4b03-81b8-640fceab02d7.png)


At first we have text data like above, then it will be encoded into several dimensions in the form of numbers. The encoded numbers will be placed in certain dimensions.
These numbers will form a collection whose contents are numbers that have similarities. it means that we get the similarity from TFIDF in the form of numbers that unite in a group, automatically we also get the similarity of the document. so behind the scenes TFIDF will look for similarities in numbers that will be represented in real life (actual data). 

If you look for similarity of news documents manually, it will be difficult. So we encode, then look for the encoded numbers that have similarities and then they will be returned as news (actual data).

so this can also be used for sentiment analysis, for example in today's news what is being discussed, then we just have to look at the keywords based on the model we made. Can also be used to see the similarity of a message with other messages. Document similarity can also be used for a recommendation system, for example, if someone likes movie A, the machine can recommend movies with the same genre or synopsis.


# Dataset

The data I use is data from the Kompas news agency in Indonesia. The data contains online news from year to year that has been collected. The data consists of 2008 rows (news), and 1 column, namely text and it's structured.

# Import Package

import **pandas as pd**

from **nltk.tokenize** import **word_tokenize**

from **nltk.corpus** import **stopwords**

from **string** import **punctuation**

from **sklearn.feature_extraction.text** import **TfidfVectorizer**

from **sklearn.metrics.pairwise** import **cosine_similarity**

# Import Data

import data with pandas. 

![Screenshot 2022-05-12 203927](https://user-images.githubusercontent.com/86812576/168088413-eb49ccf0-bf25-40eb-848c-43a73bcf4b9d.png)

# Explanation
# Extract TFIDF

We will see if it works or not, see the similarity of an encoded text. Logically it is possible, for example try to imagine in BoW (Bag of Words) it only counts words so it will count words in every document. if there are documents that have the same word counted then they are most likely have similar documents. But in this case i will use TFIDF not BoW.

![Screenshot 2022-05-18 205407](https://user-images.githubusercontent.com/86812576/169057458-277c41fb-5ce9-4897-b156-6a8a60da8190.png)

First, i will do TFIDF vectorizing, input n_gram as much as one or two words. The tokenizer uses word_tokenizer from nltk, and _stopwords_ uses stopwords made by Indonesians because this project is in Indonesian. Then just fit_transform

# TFIDF Similarity -> Document Similarity

Remember, if we have a sentence, it will be made into a vector, and the vector has its place in a certain dimension. For example, if there are only three vectors, it will have three dimensions. for now our vector is as much as the number of vocabulary so it's multi dimensional we can't imagine. so we just need to find the similarity, and the technique to find it is with cosine similarity

### What is Cosine Similarity?
![image](https://user-images.githubusercontent.com/86812576/169061574-2dfa5bd0-de6b-4b12-9492-fba0fd72890e.png)

For example, if we have two vectors (item 1, and item 2), then we only need to look at the angles, if the angles are small (adjacent) then the document is considered similar, otherwise if the angles are opposite then this is a contradicting document.

### Sorting Similar Document
I will do index sorting whose documents have similarities with other documents. For example I will take the index document 0 to see which documents are similar to it.

![srt](https://user-images.githubusercontent.com/86812576/169065113-0de07e12-d836-433a-90e0-40624944d8cd.png)
![srt2](https://user-images.githubusercontent.com/86812576/169065335-068a6283-1539-4fc3-ac47-e00ff66b07a1.png)

From the sorting results above, it can be seen that the document that is most similar to the index 0 document is index 0, of course because it is the same document. the document that has the second similarity is index 144, the third similarity is index 215.

### is it true? let's see the proof.

![srt3](https://user-images.githubusercontent.com/86812576/169065197-4afb861e-c64b-4198-8e12-41d2638f1c51.png)

We can see the results above that index documents 144, and 215 are the indexes that have the most similarity with index 0, while index 932 is the most contradictory document with index 0.

Cool, right...
only by resemblance the feature is equivalent to the resemblance of the document because of the encoding.

# Keyword Extraction
The idea is that TFIDF has already given weight to specific words, then we just focus on those specific words because that's actually the keyword, so what we do is sort it.

![kw](https://user-images.githubusercontent.com/86812576/169202647-dff07e26-a820-4f63-8c2a-a86ecdc3bddb.png)

The first thing we do is create a variable that holds all the vocabulary we have. Next is the TFIDF matrix is ​​changed to an array and then add the argsort function to get the index.

