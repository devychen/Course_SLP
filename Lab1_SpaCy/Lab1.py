import spacy
from spacy import displacy

# load the large english model
nlp = spacy.load("en_core_web_lg")

def preprocess_text1(text):
    """
    Preprocess the input text and print the resulting tokens.

    Preprocessing includes lemmatization and removal of stopwords.

    :param text: str - the text to preprocess
    :return: None
    """
    pass
    # preprocess
    doc = nlp(text)
    # lemmatisation
    preprocessed_tokens = []
    print("After removal stopwords:")
    for token in doc:
        if not token.is_stop:
            preprocessed_tokens.append(token.lemma_)
    print(preprocessed_tokens, "\n")


def preprocess_text2(text):
    """
    Preprocess the input text and print the resulting tokens.

    Preprocessing includes lemmatization, lowercasing,
    removal of: urls, email addresses, punctuation, and whitespace.

    :param text: str - the text to preprocess
    :return: None
    """
    # Preprocess
    doc = nlp(text)
    # Initialize list to store preprocessed tokens
    preprocessed_tokens = []
    print("After removal:")
    # Execute removals
    for token in doc:
        if not token.like_url and not token.like_email and not token.is_punct and not token.is_space and not token.is_stop:
            preprocessed_tokens.append(token.lemma_.lower())
    print(preprocessed_tokens, "\n")


def get_nouns(text):
    """
    Print all NOUNs in text.

    :param text: text to process
    :return: None
    """
    doc = nlp(text)
    print("Nouns: ")
    nouns = []
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
    # Or: nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    
    print(nouns, "\n")


def get_dep_roots(text):
    """
    Print the ROOT token of the dependency tree
    of each sentence in text.

    :param text: text to process
    :return: None
    """
    doc = nlp(text)
    print("Root token: ")
    for sent in doc.sents:   # attribute `doc`, property
        print(sent.root.text)


def named_entities(text):
    """
    Annotate text and print all named entities and their labels.

    :param text: text to annotate
    :return: None
    """
    doc = nlp(text)
    print("\nNamed entities: ")
    for entity in doc.ents:
        print((entity.text, entity.label_))
    print("\n")

def similarity(text1, text2):
    """
    Annotate both texts and print the similarity score.

    :param text1: first text
    :param text2: second text
    :return:
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    print("Similarity is:", doc1.similarity(doc2))


def main():
    preprocess_text1("I couldn't get home before the storm began,\nso I got soaked in the downpour.")
    preprocess_text2("There's lots of interesting Ted Talks at https://www.ted.com/.\n"
                            "For more info write to info@example.com :-)")
    get_nouns("I couldn't get home before the storm began,\nso I got soaked in the downpour.")
    get_dep_roots("I couldn't get home before the storm began. I got soaked in the downpour.")
    named_entities("Is OpenAI's ChatGPT a threat to Google? CEO Sundar Pichai made a statement today in New York.")
    similarity("I couldn't get home before the storm began.", "It's raining cats and dogs today.")
    similarity("I couldn't get home before the storm began.", "I got soaked in the downpour.")
    similarity("It's raining cats and dogs today.", "I got soaked in the downpour.")


if __name__ == '__main__':
    main()