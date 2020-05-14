# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from spacy.matcher import Matcher
import pandas as pd

#nltk.download('stopwords')


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

#python -m spacy download en

os.chdir("E:/Go Crack IT Internship")

STOPWORDS = set(stopwords.words('english'))

with open('jd1.txt', 'r') as file:
    data = file.read().replace('\n', '')
    

def preprocess(sent):
    sent = re.sub('[^a-zA-Z0-9 \n]', '', sent)
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


sent = preprocess(data)
pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)
type(cs)


from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)



ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(data)))
pprint(ne_tree)
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree

def get_continuous_chunks(text, label):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree and subtree.label() == label:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk


get_continuous_chunks(data, 'ORGANIZATION')

#nltk.download('treebank')
#from itertools import groupby
#for tag, chunk in groupby(ne_tree, lambda x:x[1]):
#    if tag != "O":
#        print("%-12s"%tag, " ".join(w for w, t in chunk))

#from nltk import Tree
#parsed_Tree = Tree('ROOT', )

#ne_tree.leaves()
#
#for ent in sent.ents:    
#    if ent.label_ in ['LOC', 'GPE']:
#        print(ent.text, ent.label_)  

##########################


#from nltk.tag import StanfordNERTagger
#import os
#
#java_path = "C:/Program Files/Java/jdk-11.0.2/bin/java.exe"
#os.environ['JAVAHOME'] = java_path #this java path you get from command 'echo %PATH%'in terminal 
#
#st = StanfordNERTagger('E:/Go Crack IT Internship/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
#                       'E:/Go Crack IT Internship/stanford-ner-2018-10-16/stanford-ner.jar',encoding='utf-8')
#
#
#result=st.tag(data.split())
#
#def get_continuous_chunks(tagged_sent):
#    continuous_chunk = []
#    current_chunk = []
#
#    for token, tag in tagged_sent:
#        if tag != "O":
#            current_chunk.append((token, tag))
#        else:
#            if current_chunk: # if the current chunk is not empty
#                continuous_chunk.append(current_chunk)
#                current_chunk = []
#    # Flush the final current_chunk into the continuous_chunk, if any.
#    if current_chunk:
#        continuous_chunk.append(current_chunk)
#    return continuous_chunk
#
#
#named_entities = get_continuous_chunks(result)
#named_entities = get_continuous_chunks(result)
#named_entities_str = [" ".join([token for token, tag in ne]) for ne in named_entities]
#named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]
#
#print(named_entities)
#print(named_entities_str)
#print(named_entities_str_tag)

###########################



#Skills

#data_refined =  re.sub('[^a-zA-Z0-9 \n]', '', data)
#data_refined = data_refined.lower().split()       # Convert to lower case
#stops = set(stopwords.words("english")) 
#meaningful_words = [w for w in data_refined if not w in stops]  
#data_refined = ( " ".join( meaningful_words ))
#
#doc = nlp(data_refined)
#
#pprint([(X.text, X.label_) for X in doc.ents])
#pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
#
#len(doc.ents)
#labels = [x.label_ for x in doc.ents]
#
#items = [x.text for x in doc.ents]
#Counter(items).most_common(3)
#
#sentences = [x for x in doc.sents]
#print(sentences[10])
#
#displacy.render(nlp(str(sentences[10])), jupyter=True, style='ent')
#
#displacy.render(nlp(str(sentences[10])), style='dep', jupyter = True, options = {'distance': 120})
#
#[(x.orth_,x.pos_, x.lemma_) for x in [y 
#                                      for y
#                                      in nlp(str(sentences[10])) 
#                                      if not y.is_stop and y.pos_ != 'PUNCT']]
#
#
#dict([(str(x), x.label_) for x in nlp(str(sentences[10])).ents])
#
#print([(x, x.ent_iob_, x.ent_type_) for x in sentences[0]])
skills_file=None
doc = nlp(data)

noun_chunks = list(doc.noun_chunks)

tokens = [token.text for token in doc if not token.is_stop]
#    if not skills_file:
#        data = pd.read_csv(
#            os.path.join(os.path.dirname(__file__), 'skills.csv')
#        )
#    else:
        
skills_data = pd.read_csv("E:/Go Crack IT Internship/Python Script/skills.csv")
skills = list(skills_data.columns.values)
skillset = []

for token in tokens:
    if token.lower() in skills:
        skillset.append(token)

    # check for bi-grams and tri-grams
for token in noun_chunks:
    token = token.text.lower().strip()
    if token in skills:
        skillset.append(token)

skills_final = [i.capitalize() for i in set([i.lower() for i in skillset])]


###############################
from nltk.stem.wordnet import WordNetLemmatizer
import re

#nltk.download('wordnet')
#resume_text = data

def extract_experience(resume_text):
    '''
    Helper function to extract experience from resume text

    :param resume_text: Plain resume text
    :return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [
            w for w in word_tokens if w not
            in stop_words and wordnet_lemmatizer.lemmatize(w)
            not in stop_words
        ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    constants = cp.parse(sent)

    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)

    test = []

    for vp in list(
        constants.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    # Search the word 'experience' in the chunk and
    # then print out the text after it
    x = [
        x[x.lower().index('experience') + 10:]
        for i, x in enumerate(test)
        if x and 'experience' in x.lower()
    ]
    
    if not x:
        mod_str = data.rpartition('yrs')[0]
        x = mod_str[-4:]
    elif not x:
        mod_str = data.rpartition('years')[0]
        x = mod_str[-4:]
        
    return x


experience = extract_experience(data)

##################################

#observations = field_extraction.extract_fields(observations)
#
#def extract_fields(df):
#    for extractor, items_of_interest in lib.get_conf('extractors').items():
#        df[extractor] = df['text'].apply(lambda x: extract_skills(x, extractor, items_of_interest))
#    return df
#
#def get_conf(conf_name):
#    return load_confs()[conf_name]
#
#def load_confs(confs_path='../confs/config.yaml'):
#    # TODO Docstring
#    global CONFS
#
#    if CONFS is None:
#        try:
#            CONFS = yaml.load(open(confs_path))
#        except IOError:
#            confs_template_path = confs_path + '.template'
#            logging.warn(
#                'Confs path: {} does not exist. Attempting to load confs template, '
#                'from path: {}'.format(confs_path, confs_template_path))
#            CONFS = yaml.load(open(confs_template_path))
#    return CONFS
#
#
#
#
#def extract_skills(resume_text, extractor, items_of_interest):
#    potential_skills_dict = dict()
#    matched_skills = set()
#
#    # TODO This skill input formatting could happen once per run, instead of once per observation.
#    for skill_input in items_of_interest:
#
#        # Format list inputs
#        if type(skill_input) is list and len(skill_input) >= 1:
#            potential_skills_dict[skill_input[0]] = skill_input
#
#        # Format string inputs
#        elif type(skill_input) is str:
#            potential_skills_dict[skill_input] = [skill_input]
#        else:
#            logging.warn('Unknown skill listing type: {}. Please format as either a single string or a list of strings'
#                         ''.format(skill_input))
#
#    for (skill_name, skill_alias_list) in potential_skills_dict.items():
#
#        skill_matches = 0
#        # Iterate through aliases
#        for skill_alias in skill_alias_list:
#            # Add the number of matches for each alias
#            skill_matches += lib.term_count(resume_text, skill_alias.lower())
#
#        # If at least one alias is found, add skill name to set of skills
#        if skill_matches > 0:
#            matched_skills.add(skill_name)
#
#    return matched_skills

############
