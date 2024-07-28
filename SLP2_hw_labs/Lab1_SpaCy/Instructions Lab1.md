# Lab 1: Intro to spaCy

In this lab, we will learn the basics of [spaCy](https://spacy.io/), a popular NLP library. We will use an English model here, but spaCy has pretrained models for many languages.

## Before you begin

1. Create a new python project and a virtual environment for it in the usual way.
2. Add the starter code in Lab1.py to your project.

### Install spaCy

Install spaCy like any other library via the standard

```
pip install spacy
```

### Download and install a spaCy model

We need a pretrained model to use for annotating text. We'll use the large-sized English model.
Download it with the following command **in the terminal**:

```
python -m spacy download en_core_web_lg
```

That's it. When you load the model in your code, spacy will know where to find it.

## Basics

Have a look at [spacy 101](https://spacy.io/usage/spacy-101) for an overview of what spacy is and how to get started. The section on [Linguistic annotations](https://spacy.io/usage/spacy-101#annotations)  provides examples of the available pipelines (tokenization, pos, ner, etc).

In the spacy documentation, the loaded model is called *nlp*, and the annotations of the model for a particular text is called *doc*, like this:

```python
import spacy

nlp = spacy.load("en_core_web_lg")

doc = nlp("This is some text to annotate. It can contain many sentences.")

# iterate over tokens in doc
for token in doc:
    print(token.text)

# process each sentence separately
for sent in doc.sents:
    for token in sent:
        print(token.text)
```

## Tasks

1. Preprocessing

Implement the preprocessing functions **preprocess_text1** and **preprocess_text2** in Lab1.py according to the instructions.
You may need to refer to the complete list of [Token attributes](https://spacy.io/api/token#attributes) to complete the task.

2. POS Tags

Implement the **get_nouns()** function to find all NOUNs in a text.

3. Dependency Parse

Implement the **get_dep_roots()** funtion to find the ROOT of each sentence in a text.

4. Named Entities

Implement the **named_entities()** funtion to find all named entities (and their labels) in a text.

5. Visualizations

Often it is helpful to visualize the annotations, which can easily be done with [displaCy](https://spacy.io/usage/visualizers), which renders visualizations that can be viewed in your browser.

Update the **get_dep_roots()** and **named_entities()** to visualize the dependency trees / named entities with displaCy.

> If you get an error about the port being already in use, add a *port=xxxx* argument to the call to displacy, where any 4-digit number > 5000 for xxxx should be fine.

6. Similarity

Implement the **similarity()** method, which calculates and prints the similarity of the two input texts.
