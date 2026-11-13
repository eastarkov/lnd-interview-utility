from collections import Counter
from functools import lru_cache
import re
import pymorphy3
from nltk.stem.snowball import SnowballStemmer

@lru_cache(maxsize=1)
def morph():
    return pymorphy3.MorphAnalyzer()

RU_STEM = SnowballStemmer('russian')
EN_STEM = SnowballStemmer('english')
STOP = set('и а но или да что чтобы как так то это этот тот он она они оно мы вы я ты его ее их мой твой наш ваш свой себя мне мне меня тебе тебя нам вам у в во на за из с со к ко от до по для о об обо при без над под перед через между после около не ни нет бы же ли вот тут там где когда если потому поэтому который кто чей какой быть был была были есть будет будут был бы б для про the a an and or but of in on at to from for with is are was were be been being it its this that these those i you we they he she my our their as by not no if then than which who'.split())

@lru_cache(maxsize=50000)
def lemma(token):
    token = token.lower().replace('ё','е')
    return morph().parse(token)[0].normal_form if re.search('[а-я]', token) else token

@lru_cache(maxsize=50000)
def family(token):
    base = lemma(token)
    # Stemming additionally joins derivations: прикол / прикольный / прикольное.
    stem = RU_STEM.stem(base) if re.search('[а-я]',base) else EN_STEM.stem(base)
    if stem in {"класс", "классн"}:
        return "класс"
    if stem.startswith("приколь") or stem == "прикол":
        return "прикол"
    return stem


def tokens(text):
    return re.findall(r'[^\W_]+', text.lower().replace('ё','е'), re.UNICODE)


def word_matches(words, phrase):
    stream, owners = [], []
    for i,w in enumerate(words):
        parts = tokens(w['word'])
        stream.extend(family(p) for p in parts)
        owners.extend([i]*len(parts))
    needle = [family(p) for p in tokens(phrase)]
    if not needle: return []
    result=[]
    for i in range(len(stream)-len(needle)+1):
        if stream[i:i+len(needle)] == needle:
            a,b=owners[i],owners[i+len(needle)-1]
            result.append({'start':words[a]['start'],'end':words[b]['end'], 'text':''.join(w['word'] for w in words[max(0,a-4):b+5])})
    return result


def frequent_words(records, limit=24):
    counts, forms = Counter(), {}
    for record in records:
        for word in record['words']:
            for token in tokens(word['word']):
                normal=lemma(token)
                if normal in STOP or len(normal)<3 or normal.isdigit(): continue
                if re.search('[а-я]',token) and morph().parse(token)[0].tag.POS in {'PREP','CONJ','PRCL','NPRO','INTJ'}: continue
                group=family(token)
                counts[group]+=1
                forms.setdefault(group,Counter())[normal]+=1
    return [(forms[g].most_common(1)[0][0], n) for g,n in counts.most_common(limit)]
