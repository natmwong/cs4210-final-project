# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS  ('not' and 'no' removed — kept for negation marking)
# ─────────────────────────────────────────────────────────────────────────────
import nltk


STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','nor','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn','game','games','play','played','playing','get','got','like',
}

# Kept out of STOPWORDS so they survive to the negation-marking step
NEGATION_TRIGGERS = frozenset({
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'cannot',
})

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────────────────────
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords as _nltk_sw
from nltk.tokenize import word_tokenize

# Remove negation triggers from NLTK stopwords so they survive filtering
NLTK_STOPWORDS = set(_nltk_sw.words('english')) - NEGATION_TRIGGERS