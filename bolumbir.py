
from nltk.stem import PorterStemmer, snowball
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
#verisetini açıyoruz
df = pd.read_json('cranfield_data.json', orient='body')
kelime=df["body"]

#verisetinin içindeki anlam ifade etmeyen kelimeleri çıkartıyoruz
docs=kelime
docs = kelime
docs = docs.map(lambda x: re.sub('[,\.!?()1234567890=;:$%&#]', '', x))
docs = docs.map(lambda x: re.sub(r'[^a-zA-Z0-9.\s]', ' ', x))  # Replace all non-alphanumeric characters with space
docs = docs.map(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))  # Remove all dots, e.g. U.S.A. becomes USA
docs = docs.map(lambda x: x.lower())
docs = docs.map(lambda x: x.strip())
#tokenize ettiğimiz kelimeleri eklemek için bir liste oluşturuyoruz ve tokenization işlemi yapıyoruz ardından toplama kelime sayısını yazdırıyorum
kelimeler=[]

for c in docs:
    for sent in sent_tokenize(c):
        word_tokens = word_tokenize(sent)
        kelimeler += word_tokens
print("*****************************************************************************************************")
print("Toplam kelime sayısı: ")
print(len(kelimeler))
print("*****************************************************************************************************")
#stopwordsleri fileter
filtered_words=[]
stwords = set(stopwords.words('english'))
for w in kelimeler:
    if w not in stwords:
        filtered_words.append(w)


final_filtered_words = list(set(filtered_words))
final_filtered_words.sort()

#porter ve lemmatizor u ekliyorum ve sıralayıp en son adlı listeye ekliyorum anlam ifade etmeyen bazı şeylerden kurtulmak için

stemmer = snowball.SnowballStemmer('english')
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

en_son=[]
for j in final_filtered_words:
    if len(j) == 1 or len(j) == 2:
        continue
    en_son.append(j)



en_son.sort()

print("{0:20}{1:20}{2:20}".format("Words","Porter Stemmer","Lemmatize"))
for i in en_son:
    print("{0:20}{1:20}{2:20}".format(i, porter.stem(i),lemmatizer.lemmatize(i)))



print("*****************************************************************************************************")
inverted_index = defaultdict(set)

# kelimeler üzerinde inverted index yapısı kurma
for docid, c in enumerate(filtered_words):
    for sent in sent_tokenize(c):
        for word in word_tokenize(sent):
            word_lower = word.lower()
            if word_lower not in stwords:
                word_stem = stemmer.stem(word_lower)

                inverted_index[word_stem].add(docid)


#kelimenin indexini arama
def process_and_search(query):
    matched_documents = set()
    for word in word_tokenize(query):
        word_lower = word.lower()
        if word_lower not in stwords:
            word_stem = stemmer.stem(word_lower)
            matches = inverted_index.get(word_stem)
            if matches:
                matched_documents |= matches
    return matched_documents
print("Aranan kelimenin bulunduğu indeksler:")
print(process_and_search("frozen"))
print("*****************************************************************************************************")
def _create_frequency_matrix(filtred_words):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in filtred_words:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix
print("\n\n")
print("Verisetinin içinde geçen kelimelerin terim sıklığı:")
frequency = collections.Counter(filtered_words)
print("\n\n")
print(frequency)