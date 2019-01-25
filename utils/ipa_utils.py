from espeakng import ESpeakNG
import re
import pandas as pd
import numpy as np

from vocab_utils import SOS, EOS


__all__ = [
    'IPAError',
    'get_ipa',
    'load_binf2phone'
]


class IPAError(ValueError):
    pass


def _postprocessing(ipa):
    # remove language switch markers
    ipa = re.sub(r'(\([^)]+\))', '', ipa)
    # remove diacritics
    ipa = ''.join(x for x in ipa if x.isalnum())
    return ipa


def get_ipa(text, language):
    engine = ESpeakNG()
    engine.voice = language.lower()
    ipa = engine.g2p(text, ipa=2)
    if ipa.startswith('Error:'):
        raise IPAError(ipa)
    return _postprocessing(ipa)

def load_binf2phone(filename):
    binf2phone = pd.read_csv(filename, index_col=0)
    binf2phone.insert(0, EOS, 0)
    binf2phone.insert(0, SOS, 0)
    bottom_df = pd.DataFrame(np.zeros([2, binf2phone.shape[1]]),
                             columns=binf2phone.columns, index=[SOS, EOS])
    binf2phone = pd.concat((binf2phone, bottom_df))
    binf2phone.loc[binf2phone.index==SOS, SOS] = 1
    binf2phone.loc[binf2phone.index==EOS, EOS] = 1
    return binf2phone

def ipa2binf(ipa, binf2phone):
    pass