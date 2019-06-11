# author : Yurdha Fadhila (yurdhafadhila@gmail.com)

import sklearn
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import numpy
import re


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy as sp
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
# from spacy.tokenizer import Tokenizer
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')

corpus = open("short-corpus.txt", "r")
korpus = corpus.read()
sentences = korpus.split('. ')


def find_kandidat_jawaban(question):
   
    #frequency
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(sentences)
    freq_term_corpus = X.toarray()
    freq_term_question = vectorizer.transform(question).toarray()
   
    
    #tf idf
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf_corpus = transformer.fit_transform(freq_term_corpus)
    tfidf_question = transformer.fit_transform(freq_term_question)   

    #cosine
    list_cosine =[]
    for tc in tfidf_corpus:
        temp = cosine_similarity(tc, tfidf_question)
        list_cosine.append(temp[0,0])

    if max(list_cosine) != 0.0 :
        index_max = list_cosine.index(max(list_cosine))
        kandidat_jawaban = sentences[index_max]
    else:
        kandidat_jawaban = 'none'
    
    return kandidat_jawaban

# this function bellow code by @allysas
def getPOSTagging(sentence):
    listOfSentence_tokens = []
    listOfSentence_tagged = []  
    sentence_tokens = nltk.word_tokenize(sentence)
    listOfSentence_tokens += sentence_tokens
    sentence_tagged = nltk.pos_tag(sentence_tokens)
    listOfSentence_tagged += sentence_tagged

    return(listOfSentence_tokens, listOfSentence_tagged)

# this function bellow code by @allysas
def getIOBNER(sentence):
    listOfSentence_NER = []
    document = nlp(sentence)
    temp = [(X, X.ent_iob_, X.ent_type_) for X in document]
    listOfSentence_NER += temp
    return listOfSentence_NER

def no_answer():
    print('sorry, i cant find what you\'re looking for')

def find_answer_type(kandidat_jawaban, question):
    hasil_tokens_kandidat_jawaban, hasil_tagged_kandidat_jawaban = getPOSTagging(kandidat_jawaban)
    hasil_oib_kandidat_jawaban = getIOBNER(kandidat_jawaban)
    hasil_tokens_question, hasil_tagged_question = getPOSTagging(question)
    hasil_oib_question = getIOBNER(question)
   
    question_type = hasil_tagged_question[0]
    question_type2 = hasil_tagged_question[1]

    if question_type[0].lower() == 'who' and question_type2[0].lower() == 'is':

        if  [item for item in hasil_oib_question if 'PERSON' in item[2]] != []:
            query = [str(item[0]) for item in hasil_oib_question if 'PERSON' in item[2]]
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'ORG'
                # print(question_looking_for)
            else:
               question_looking_for = 'none'
        elif [item for item in hasil_oib_question if 'ORG' in item[2]] != []:
            query = [str(item[0]) for item in hasil_oib_question if 'ORG' in item[2]]
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'PERSON'
            else:
               question_looking_for = 'none'            
        else:
            question_looking_for = 'none'

    elif question_type[0].lower() == 'what' and question_type2[0].lower() == 'time':
        query = [item[0] for item in hasil_tagged_question if 'NN' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'TIME'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'
            

    elif question_type[0].lower() == 'how' and question_type2[0].lower() == 'much':
        query = [item for item in hasil_tagged_question if 'NN' in item]
        if [query2 for query2 in query if 'money' in query2] != []:
            question_looking_for = 'MONEY'
            # print(question_looking_for)
        else:
            question_looking_for = 'none'
            # print(question_looking_for)

    elif question_type[0].lower() == 'how' and question_type2[0].lower() == 'many':
        query = [item[0] for item in hasil_tagged_question if 'NNS' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'QUANTITY'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'

    elif question_type[0].lower() == 'how' and question_type2[0].lower() == 'big':
        query = [item[0] for item in hasil_tagged_question if 'NN' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'QUANTITY'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'

    elif question_type[0].lower() == 'how' and question_type2[0].lower() == 'tall':
        query = [item[0] for item in hasil_tagged_question if 'NNS' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'QUANTITY'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'

    elif question_type[0].lower() == 'when':
        query = [item[0] for item in hasil_tagged_question if 'NN' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'DATE'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'

    elif question_type[0].lower() == 'where':
        query = [item[0] for item in hasil_tagged_question if 'NN' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'LOCATION'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'

    elif question_type[0].lower() == 'what':
        query = [item[0] for item in hasil_tagged_question if 'NNP' in item]
        if query != []:
            if [check for check in query if check in hasil_tokens_kandidat_jawaban] != []:
                question_looking_for = 'DESCRIPTION'
                # print(question_looking_for)
            else:
                question_looking_for = 'none'
        else:
            question_looking_for = 'none'
    else:
        question_looking_for = 'none'

    return question_looking_for        

def get_answer(question, kandidat_jawaban, answer_type):
    hasil_tokens_kandidat_jawaban, hasil_tagged_kandidat_jawaban = getPOSTagging(kandidat_jawaban)
    hasil_oib_kandidat_jawaban = getIOBNER(kandidat_jawaban)
    hasil_tokens_question, hasil_tagged_question = getPOSTagging(question)
    hasil_oib_question = getIOBNER(question)
    
    if answer_type is 'ORG':
        answer = [item[0] for item in hasil_oib_kandidat_jawaban if 'ORG' in item[2] ]
        stringg = ''
        for a in answer:
            stringg += str(a)
            stringg += ' '
    elif answer_type is 'PERSON':
        answer = [item[0] for item in hasil_oib_kandidat_jawaban if 'PERSON' in item[2] ]
        stringg = ''
        for a in answer:
            stringg += str(a)
            stringg += ' '
    elif answer_type is 'DATE':
        answer = [item[0] for item in hasil_oib_kandidat_jawaban if 'DATE' in item[2] ]
        stringg = ''
        for a in answer:
            stringg += str(a)
            stringg += ' '
        # print(stringg)
    elif answer_type is 'TIME':
        pattern = r"\d{2}:\d{2}\s[A-Z]{4}"
        answer = re.findall(pattern, kandidat_jawaban)
        stringg = ''
        for a in answer:
            stringg += str(a)
            stringg += ' '
        # print(stringg)
    elif answer_type is 'MONEY':
        answer = [item[0] for item in hasil_oib_kandidat_jawaban if 'MONEY' in item[2] ]
        stringg = ''
        for a in answer:
            stringg += str(a)
            stringg += ' '
        # print(stringg)
    elif answer_type is 'QUANTITY':
        query = [item for item in hasil_tagged_question if 'NNS' in item[1]]
        stringg = ''
        if query != []:
            # print(query)
            index_query = hasil_tagged_kandidat_jawaban.index(query[0])
            
            temp = []
            temp.append(hasil_tagged_kandidat_jawaban[index_query-1])
            temp.append(hasil_tagged_kandidat_jawaban[index_query-2])
            answer = [item[0] for item in temp if 'CD' in item[1]]

            if len(answer) < 1:
                answer = [item[0] for item in hasil_oib_kandidat_jawaban if 'QUANTITY' in item[2]]
            # print(answer)
            # stringg = ''
            for a in answer:
                stringg += str(a)
                stringg += ' '
        # print(stringg)
    elif answer_type is 'LOCATION':
        answer = [item[0] for item in hasil_oib_kandidat_jawaban if 'GPE' in item[2] ]
        stringg = ''
        for a in answer:
            stringg += str(a)
            stringg += ' '
        # print(stringg)
    elif answer_type is 'DESCRIPTION':
        stringg = kandidat_jawaban
    
    return stringg

def question_answer(user_question):
    list_question = []
    list_question.append(user_question)
    kandidat_jawaban = find_kandidat_jawaban(list_question)
    # print(kandidat_jawaban)
    if kandidat_jawaban == 'none':
        no_answer()
        print("\n")
    else:
        answer_type = find_answer_type(kandidat_jawaban, user_question)
        # print(answer_type)
        if answer_type == 'none':
            no_answer()
            print("\n")
        else:
            final_result = get_answer(user_question, kandidat_jawaban, answer_type)
            if len(final_result) <=1:
                no_answer()
                print("\n")
            else:
                print(final_result)
                print("\n")

# -------------------SINI
user_question = "When did the government stop the search and rescue operation?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "Who is Mohamad Nasir?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "How many people dead by tsunami?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "How many tents send from Turkish Government?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "What did the Turkish Government do?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "Where did the earthquake happened? "
print('Question : '+ user_question)
question_answer(user_question)

user_question = "What time tsunami warning was issued in Palu?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "How much money did Google donated to the victims?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "How tall were the tsunami waves?"
print('Question : '+ user_question)
question_answer(user_question)

user_question = "Who is the CEO of Google?"
print('Question : '+ user_question)
question_answer(user_question)

# user_question = "where did the large earthquake struck?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'how big the magnitude of earthquake?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'what time tsunami warning was issued in Palu?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'How much money did Google donated to the victims?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'How many Hercules send from Turkish Government?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'Who is the CEO of Google?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'What did Sundar Pichai do?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'when did the government stop the search and rescue operation?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'what did the Governor do to commemorate the victims?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'how tall were the tsunami waves?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "who is Longki Djanggola?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = 'How many people dead by tsunami?'
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "Who is the Ministry of Research, Technology, and Higher Education?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "How much money did WhatsApp donated to the victims?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "How many sleeping bags send from Turkish Government?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "How many tents send from Turkish Government?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "What did the Turkish Government do?"
# print('Question : '+ user_question)
# question_answer(user_question)

# user_question = "What did the Spanish Government do?"
# print('Question : '+ user_question)
# question_answer(user_question)


user_question = input("Question : ")
question_answer(user_question)
