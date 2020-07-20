import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import linear_kernel
dbase = pd.read_csv("final_data.csv")
dbase.rename(columns=lambda x: x.replace('NomPrenom', 'nom_prenom'), inplace=True)
print(dbase.shape)
#data_coach.drop(['product_url','pid','image','is_FK_Advantage_product','product_rating','crawl_timestamp','product_category_tree','discounted_price','overall_rating','product_specifications'],axis=1,inplace=True)
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)
def make_lower_case(text):
    return text.lower()
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text
def item(id):
    return dbase.loc[dbase['numero'] == id]['nom_prenom'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.
def recommend(item_id, num):
    print("Recommendé " + str(num) + " profil similaire à " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("profil recommandé: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

dbase['cleaned_desc'] = dbase['description'].apply(_removeNonAscii)
dbase['cleaned_desc'] = dbase.cleaned_desc.apply(func = make_lower_case)
dbase['cleaned_desc'] = dbase.cleaned_desc.apply(func = remove_stop_words)
dbase['cleaned_desc'] =dbase.cleaned_desc.apply(func=remove_punctuation)
dbase['word_count'] = dbase['cleaned_desc'].apply(lambda x: len(str(x).split()))
for  i in range (5):
  cdesc=dbase.loc[i,'cleaned_desc']
#print (cdesc)
  cdesc = cdesc.lower()
    #print(desc)
  (word_tokenize(cdesc))
  word_tokens = word_tokenize(cdesc)
#print(word_tokens)
  test_list =[]
  for i in word_tokens : 
    if i not in test_list: 
      test_list.append(i)
#print(test_list)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(dbase['cleaned_desc'])
dbase.rename(columns=lambda x: x.replace('uniq_id', 'id'), inplace=True)
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in dbase.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], dbase['numero'][i]) for i in similar_indices]

    results[row['numero']] = similar_items[1:]
    
print('similarité ok')


recommend(item_id='+86 456 570 9721', num=3)


