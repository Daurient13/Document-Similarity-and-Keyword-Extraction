# Document Similarity and Keyword Extraction

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

# Import Data

import data with pandas. 

![Screenshot 2022-05-12 203927](https://user-images.githubusercontent.com/86812576/168088413-eb49ccf0-bf25-40eb-848c-43a73bcf4b9d.png)

# Explanation
# Extract TFIDF

We will see if it works or not, see the similarity of an encoded text. Logically it is possible, for example try to imagine in BoW (Bag of Words) it only counts words so it will count words in every document. if there are documents that have the same word counted then they are most likely have similar documents. But in this case i will use TFIDF not BoW.





