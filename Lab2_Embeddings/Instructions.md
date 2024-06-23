## Lab 2: word2vec and fasttext embeddings
___

### Before you begin

1. You should have already completed the PreLab Tasks (setting up your project and downloading the embeddings)
2. Add the starter code in Lab2.py to your project.


### Loading Embeddings

The embeddings are loaded differently depending on which ones you use. The load_embeddings() function will load the embeddings given the embedding type (GOOGLE or FASTTEXT_DE_MINI).

Run the starter code in Lab2.py, which for now just loads the Google embeddings.

If you get an error, you may need to move the GoogleNews vectors file to the same directory as Lab2.py, or adjust the file path for loading.

>Notice that by setting a limit we load only the first 200,000 embeddings. Loading all 3 million embeddings takes a long time, and since the words are ordered from most to least frequent, the vocabulary should be large enough to achieve the goals of this lab.

### Basics
1. Accessing Vectors
       
   The _emb_ object can be used like a python dictionary to access embeddings:

```python
# print the vector for 'flower' if there is one
if 'flower' in emb:
    print(emb['flower'])
```

2. Find Similar Words

    Use _emb.most_similar()_ to get a list of (word, score) tuples:

```python
# get the top 5 most similar words to flower
similar = emb.most_similar('flower', topn=5)
```

3. Similarity of 2 Words
    
    Use _emb.similarity()_ to get the cosine similarity of 2 words:

```python
# get the cosine similarity of 'flower' and 'bee'
score = emb.similarity('flower', 'bee')
```

## Part 1: GoogleNews Embeddings for English

Complete following tasks using the GoogleNews embeddings (English).

## Tasks (Part 1)

### Explore the Words

Complete the _prompt\_vector()_ function according to the instructions, then answer the following questions:

- Are the words case-sensitive? lemmatized?
- Are contractions (e.g. "don't", "can't", "I'd") included? What about "n't"?
- Are punctuation marks, currency symbols, etc. included? What about digits/numbers?

### Most Similar Words
    
Complete the _prompt\_most\_similar()_ function according to the instructions, then answer the following questions:

- How would you describe the similarity?
- What about words with multiple senses, like 'bank' or 'mouse'?

    > Word embeddings are highly dimensional, and each dimension captures some feature of a word. There are numerous methods for reducing the dimension space, which allow us to plot words as points on a 2D graph. Different dimensionality reduction techniques preserve different aspects of the data. Here we use a popular method called T-SNE (t-distributed stochastic neighbourhood embedding).

### Cosine Similarity of 2 Words

Complete the _prompt\_similarity()_ function according to the instructions.

- What similarity scores do you get for word pairs that are synonyms?
- What similarity scores do you get for word pairs that are not synonyms (e.g. antonyms)
- What similarity scores do you get for word pairs that are not related at all?

## Part 2: FastText Embeddings

FastText embeddings are trained with token n-grams, rather than full tokens, which allow them to capture meaning of subwords, and makes them useful when working with morphologically rich languages like German.

See this post for a [short description of the intuition behind FastText](https://amitness.com/2020/06/fasttext-embeddings/).

> The original FastText embeddings are available for 157 languages, and come in 2 forms:
> - **binary**: contain everything needed to instantiate the model and resume training. Binary versions have the nice property that they can generate embeddings for OOV words by combining the vectors of the OOVs subwords (token n-grams). These are very large and not suitable if memory is limited.
> - **text**: static vectors with a fixed vocabulary, so embeddings are not generated for OOV words. These are smaller than the binary models.
> 
> **Compressed FastText Binaries** are compressed versions of the original binary embeddings. They are quite small and particularly well suited for applications with limited memory resources. Since they are binary versions, they also use subwords to generate embeddings for OOV words.

## Tasks (Part 2)

Complete following tasks using the FastText compressed binary embeddings (German).

### OOV words

Consider the German compound _Kindergeburtstagskuchen_, which is not contained in the FastText static vectors.
The compressed embeddings construct a vector for _Kindergeburtstagskuchen_, from its subwords.

Run your code again using the German compressed binary embeddings, then answer the following questions:

- What is the embedding for _Kindergeburtstagskuchen_?
- If an embedding can't be constructed, a vector of zeros is returned. Print the vector for _abcdefg_.
- Which words are most similar to _Kindergeburtstagskuchen_?
- Which words are most similar to _Tischtennisschiedsrichterin_?
- What is the similarity score between _Kindergeburtstagskuchen_ and _Kindergeburtstagsgeschenk_?
