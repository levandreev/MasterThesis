# a pipeline for NLP that does: part of speech tagging,lemmatisation, entity recognition


import spacy
# exclude_from_pipeline = [] # e.g. ['tagger', 'parser', 'ner']
# lemmatization = True
# nlp = spacy.load('en_core_web_sm',disable=exclude_from_pipeline) # Language Model with option of disabling pipeline components
# # nlp = spacy.load('en_core_web_sm')
#
# print('pipeline components: ', nlp.pipe_names)

# spaCy first tokenizes the text to produce a Doc object
# doc = nlp('Apple is looking at buying U.K. startup for $1 billion Barrack Obama')
# doc = nlp('Yahoo! developed an application that collects the most-read news stories from different categories for iOS and Android.')
#
# if lemmatization:
#     print('\nLemmatization:')
#     for token in doc:
#         print(token, '---> ', token.lemma_)
#
# if 'ner' in nlp.pipe_names:
#     print('\nNamed Entity Recognition:')
#     for ent in doc.ents:
#         print(ent, '--->', ent.label_)
#
# if 'tagger' in nlp.pipe_names:
#     print('\nPOS Tagging:')
#     for token in doc:
#         print(token, '---> ', token.tag_)

def process_document(text, lemmatization, tagger, parser, ner):
    """A function with options
    that you can flag on or off
    """

    exclude_from_pipeline = []
    if not tagger:
        exclude_from_pipeline.append('tagger')
    if not parser:
        exclude_from_pipeline.append('parser')
    if not ner:
        exclude_from_pipeline.append('ner')
    nlp = spacy.load('en_core_web_sm', disable=exclude_from_pipeline)
    print('pipeline components: ', nlp.pipe_names)
    doc = nlp(text)

    if lemmatization:
        print('\nLemmatization:')
        for token in doc:
            print(token, '---> ', token.lemma_)

    if 'ner' in nlp.pipe_names:
        print('\nNamed Entity Recognition:')
        for ent in doc.ents:
            print(ent, '--->', ent.label_)

    if 'tagger' in nlp.pipe_names:
        print('\nPOS Tagging:')
        for token in doc:
            print(token, '---> ', token.tag_)

process_document("Barrack Obama likes apples but Apple is a company", lemmatization=True, tagger=True, parser=False, ner=True)

