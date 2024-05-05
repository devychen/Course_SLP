[Return to contents](https://github.com/devychen/Notes-SNLP/tree/main#readme)

# Embedding
## Vector semantics
- **Embedding**(s): representations of meaning of words. 
  - Long & spares vectors: tf-idf, PPMI (models);
  - Short & Dense vectors: word2vec (models).
- **Vector semantics** instantiates the linguistic **distributional hypothesis** 
(words that occur in similar contexts tend to have similar meanings) by learning embeddings. 
  - Using vector semantics to represent _documents_ and/or _word meaning_ in the vector space. <br>
- Build the models of meaning based on **co-occurrence matrix** (a way of representing how often words co-occur):
  - **the term-document matrix**: 
    - Row vectors for word meaning, columns for documents. 
    - Similar documents/words - similar vectors.
    - Has $|V|$ rows (each word type), $D$ columns (each doc).
  - **the term-term matrix**
    - $|V|\times|V|$ dimensionality, each $|V|$ is the vocab size. 
    - Each cell records the number of times 
    the row word (target) and the column word (context) co-occur;
    - Context could be a document, or a window around the word.

## Computing words similarity
- Issue with **raw cosine**: it favors long vectors, resulting in high raw dot 
product for frequent words, but we want to know the similarity regardless of frequency.
- Cosine similarity metric (normalised dot product)
  - $cosine(v,w)=\frac{v\times w}{|v||w|}$
- The more similar, the smaller the angle, the larger the **cosine**.
- Range from 0-1.
- The cosine has it max when 
  - the angle is smallest i.e. 0.
  - the cosine of all other angles is less than 1.

  
## Weighting cells (weighting words similarity)
- Co-occurrence matrix represent each cell by frequency. Raw frequency is skewed
and not very discriminative (e.g. _they_ cross docs).
- 2 solutions to solve the paradox: tf-idf weighting when dimensions are docs,
PPMI algorithm when dimensions are words.
### tf-idf (product of two terms)
- **tf** term freq.: use $log(raw\ count)+1$ if $count>0$, otherwise 0.
- **idf** inverse doc freq.: use log $log_{10}\left(\frac{N}{df_t}\right)$
  - The fewer the docs in which a term occurs, the higher this weight.
- The final equation: $w_{t,d} = tf_{t,d}\times idf_t$
### PPMI
- Positive Pointwise Mutual Information.
- Intuition: how much more the two words co-occur than priori expectation to
appear by chance.
- **PMI**:
  - A measure of how often 2 events $x$ and $y$ (here the target word $w$ and the context word $c$)
  occur, compared with the expected if they are independent.
  - $PMI(w,c)=log_2\frac{P(w,c)}{P(w)P(c)}$
    - Numerator: how often we _observed_ the 2 words together
    - Denominator: how often we _would expect_ ...
- the PPMI value of word $w$ given context $c$: $PPMI(w,c)=max\left(log_2\frac{P(w,c)}{P(w)P(c)},0\right)$
- With PPMI, negative value of PMI is replaced by 0.
- Problem: bias towards infrequent events.
  - Solution 1: using a different function $P_\alpha(c)$ with a setting of $\alpha = 0.75$.
  It increases the probability assigned to rare contexts:
    - $PPMI_\alpha(w,c)=max\left(log_2\frac{P(w,c)}{P(w)P_\alpha(c)},0\right)$
    - $P_\alpha\(c)=\frac{count(c)^\alpha}{\sum_ccount(c)^\alpha}$
  - Solution 2: Laplace smoothing: before PMI, add a small constant k (usually 0.1~3)
  to each count, discounting all the non-zero values.

### Word2vec
- Embeddings are short: dimensions d ranging from 50-1000 instead of large vocab size $|V|$ or $D$; 
Vectos are dense: values are real-valued numbers that can be negative, instead of mostly-zero counts.
- Better than the sparsed: learn far fewer weights, smaller parameter space helps with generalisation and avoiding overfitting.
- Computing method: **skip-gram with negative sampling (SGNS)** in a software pkg called word2vec.
- Word2vec embeddings are static embeddings, i.e. the model learns one fixed embedding for each word in the vocab.
(On the contrary to the dynamic contextual embeddings like BERT)
- Intuition: 

#### Skip-gram (a Word2vec algorithms)
- Intuitions:

## Other Static Embeddings (Models) <br>
- **Fattest** (An extension of word2vec that solve 2 issues:
- **GloVe**

## Semantic Properties of Embeddings
- Different types of similarity or association 
  - The Parameter of context window size: the choice depends on the goals of the representation.
    - Shorter - semantically similar words within same POS; (e.g. Hogwarts, Sunnydale, Evernight - fictional schools in other fictions)
    - Longer - topically related but not similar; (e.g. Harry Potter, Dumbledore, Half-blood)
  - Two kinds:
    - **First-order co-occurrence**: typically nearby each other (e.g. wrote, book/poem)
      - also called syntagmatic association.
    - **Second-order co-occurrence**: have similar neighbors (e.g. wrote, said)
      - also called paradigmatic ~.

## Other Related Terms
- **Information retrieval (IR)**: the taks of finding the doc $d$
from the $D$ docs in some collection that best matches a query $q$.
- **Vectors**: at heart, just a list/array of numbers.

[Return to contents](https://github.com/devychen/Notes-SNLP/tree/main#readme)