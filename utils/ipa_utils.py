from espeakng import ESpeakNG
import re


__all__ = [
    'IPAError',
    'get_ipa'
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
