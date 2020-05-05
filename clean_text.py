import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

STOP_WORDS = set(stopwords.words('english'))
STOP_SYMBOLS = '.,!?:;-\n\r()/=+*"' + "'"


def delete_bad_string(text: str):
    """
    Delete wrong lines from text of whitepaper

    Parameters
    -----------
    text:
         Text of whitepaper

    Returns
    -----------
    str:
       Clean text of whitepaper
    """
    text = re.sub('[\f\t(\uf101)●➔·•]', '', text)  # delete wrong symbols
    text = re.sub('\n{2,}', '\n', text)  # repeat \n
    text = re.sub('( +)\n', '', text)  # empty string
    text = re.sub(',{2,}', ',', text)  # repeat ,
    text = re.sub(' {2,}', ' ', text)  # repeat space
    text = re.sub('-{2,}', '-', text)  # repeat -
    a = text.split('\n')
    a = [item.strip() for item in a if len(item) > 10]  # delete string if len is less then 10 symbols
    a = [item for item in a if not re.search('^([Ff]igure|[Tt]able)', item)]  # strings without Figure and Table
    a = [item for item in a if not re.search('([.…]( ?)(\d+)$)', item)]  # delete content
    a = [item for item in a if not re.search('\d', item)]
    return ' '.join(a)


def canonize_words(source):
    """
    Remove punctuation and stopwords from text

    Parameters
    ----------
    source
        text for remove

    Returns
    ----------
    list
        remaining words

    """
    return [x for x in [re.sub(r'[\.,!?:\-;\n\r()/=\+\*&#%$@\'\"●•·➔]', '', y) for y in source.lower().split()] if
            x and (x not in STOP_WORDS)]


def clean_text(text: str):
    """
    Return clean text

    Parameters
    ----------
    text
        text for clean

    Returns
    ----------
    list
        clean words
    """
    return canonize_words(delete_bad_string(text))
