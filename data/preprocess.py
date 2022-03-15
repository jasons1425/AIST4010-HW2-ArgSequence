from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import re


TOKEN_DICTIONARY = {c: idx+1 for idx, c in enumerate("ABRNDCEQGHILKMFPSTWYVXZ")}
TARGET_CLASSES = ["sp", "aminoglycoside", "macrolide-lincosamide-streptogramin",
                  "polymyxin", "fosfomycin", "trimethoprim", "bacitracin",
                  "quinolone", "multidrug", "chloramphenicol","tetracycline",
                  "rifampin", "beta_lactam", "sulfonamide", "glycopeptide"]
TARGET_DICTIONARY = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


def _get_tokenizer(token_dicts=TOKEN_DICTIONARY, char_level=True):
    tokenizer = Tokenizer(num_words=len(token_dicts), char_level=char_level)
    tokenizer.fit_on_texts(list(token_dicts))
    return tokenizer


seq_tokenizer = _get_tokenizer(TOKEN_DICTIONARY, char_level=True)


def label_encode(seq, tokenizer=seq_tokenizer, pad_len=None):
    encoded = tokenizer.texts_to_sequences(list(seq))
    if pad_len:
        encoded = pad_sequences(encoded, padding='post', truncating='post',
                                maxlen=pad_len, value=0)
    return encoded


def seq2ohe(seq, tokenizer=seq_tokenizer, pad_len=None):
    encoded = label_encode(seq, tokenizer, pad_len)
    encoded = to_categorical(encoded)
    return encoded


def bert_pt_preprocess(seq, pad_len=100):
    def seq_preprocess(x, pad_len=100):
        seq_len = len(x)
        if seq_len < pad_len:
            x = x + 'X' * (pad_len - seq_len)
        x = x[:pad_len]  # only use the first pad_len characters
        x = ' '.join(list(x))
        x = re.sub(r"[UZOB]", "X", x)
        return x
