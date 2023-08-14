import zipfile
import numpy as np
from pandas import DataFrame
import math
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import OrderedDict
import os



stop_words = set(stopwords.words('english'))





param_0 = []  #HAM
param_1 = []  #SPAM
p = 0
SPAM_length = 0
HAM_length  = 0
Dict = {}





count = 0
path_ham = os.path.join(os.getcwd(), 'dataset\\ham')
path_spam = os.path.join(os.getcwd(),'dataset\\spam')
path_test = os.path.join(os.getcwd(), 'test')



os.chdir(path_ham)
HAM = os.listdir()
HAM_length = len(HAM)
for i in range(len(HAM)):
  f = open(HAM[i], 'r',encoding = 'unicode_escape')
  file_content = f.read()
  word_tokens = word_tokenize(file_content)
  for word in word_tokens:
    word = word.lower()
    if word not in stop_words:
      if word.isalpha():
        if Dict.get(word) is None:
          Dict[word] = count 
          count += 1





os.chdir(path_spam)
SPAM = os.listdir()
SPAM_length = len(SPAM)
for i in range(len(SPAM)):
  f = open(SPAM[i], 'r',encoding = 'unicode_escape')
  file_content = f.read()
  word_tokens = word_tokenize(file_content)
  for word in word_tokens:
    word = word.lower()
    if word not in stop_words:
      if word.isalpha():
        if Dict.get(word) is None:
          Dict[word] = count 
          count += 1






for i in range(len(Dict)):
  param_0.append(0)

for i in range(len(Dict)):
  param_1.append(0)




os.chdir(path_ham)
HAM = os.listdir()
for i in range(len(HAM)):
  f = open(HAM[i], 'r',encoding = 'unicode_escape')
  file_content = f.read()
  word_tokens = set(word_tokenize(file_content))
  for word in word_tokens:
    word = word.lower()
    if word not in stop_words:
      if word.isalpha():
        try:
          param_0[Dict[word]] += 1
        except KeyError:
            pass


os.chdir(path_spam)
SPAM = os.listdir()
for i in range(len(SPAM)):
  f = open(SPAM[i], 'r',encoding = 'unicode_escape')
  file_content = f.read()
  word_tokens = set(word_tokenize(file_content))
  for word in word_tokens:
    word = word.lower()
    if word not in stop_words:
      if word.isalpha():
          try:
            param_1[Dict[word]] += 1
          except KeyError:
            pass





p = (SPAM_length + 1) / (SPAM_length + HAM_length + 2)
for i in range(len(param_1)):
  param_1[i] += 1
for i in range(len(param_0)):
  param_0[i] += 1
 # total_spam = 0
 # for i in range(len(Dict)):
 #   total_spam += param_1[i]

 # total_ham = 0
 # for i in range(len(Dict)):
 #   total_ham += param_0[i]
for i in range(len(Dict)):
  param_1[i] = (param_1[i])/(SPAM_length + 1) 
for i in range(len(Dict)):
  param_0[i] = (param_0[i])/(HAM_length + 1)





def vectorize(x_test):
  word_tokens = word_tokenize(x_test)
  test_vector = []
  for i in range(len(param_0)):
    test_vector.append(0)
  for word in word_tokens:
    word = word.lower()
    if Dict.get(word) is not None:
      test_vector[Dict.get(word)] = 1
  x_test = test_vector
  return x_test




def predict(x_test):
  res = 0
  for i in range(len(Dict)):
    num = param_1[i] * (1 - param_0[i])
    den = param_0[i] * (1 - param_1[i])
#     # print(num)
#     # print(den)
    div = num / den
#     # print(str(i) + "i")
#     # print(div)
    log_div = math.log(div)
     # print(log_div)
    res = res + x_test[i] * log_div
    num = 1 - param_1[i]
    den = 1 - param_0[i]
    div = num / den
    log_div = math.log(div)
    res = res + log_div
    num = p
    den = 1 - p
    div = num / den 
    log_div = math.log(div)
    res = res + log_div
  if res > 0:
    return 0 #spam
  else:
    return 1 #non spam




os.chdir(path_ham)
accuracy = 0
HAM = os.listdir()
for i in range(len(HAM)):
  f = open(HAM[i], 'r',encoding = 'unicode_escape')
  file_content = f.read()
  x_test = vectorize(file_content)
  if predict(x_test) == 0:
    accuracy += 1

os.chdir(path_spam)
SPAM = os.listdir()
for i in range(len(SPAM)):
  f = open(SPAM[i], 'r',encoding = 'unicode_escape')
  file_content = f.read()
  x_test = vectorize(file_content)
  if predict(x_test) == 1:
    accuracy += 1


# print(accuracy/(SPAM_length + HAM_length + 2))




os.chdir(path_test)
TEST = os.listdir()
for i in range(len(TEST)):
  f = open(TEST[i],'r',encoding = 'unicode_escape',errors = 'ignore')
  file_content = f.read()
  x_test = vectorize(file_content)
  if predict(x_test) == 0:
    print(0)
  else:
    print(1)

