from text import cmudict


_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_punctuation = '!\'(),.:;?'
_space = ' '
_pad = '_'
_special = '-'

_arpabet = ['@' + s for s in cmudict.valid_symbols]

symbols = [_pad] + [_special] + list(_punctuation) + [_space] + list(_letters) + _arpabet

_JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])
_VAILD_JAMO = [jamo for jamo in _JAMO_LEADS + _JAMO_VOWELS + _JAMO_TAILS]

korean_symbol = [_pad] + [_special] + list(_punctuation) + [_space] + _VAILD_JAMO

if __name__ == '__main__':
    print(korean_symbol)
    print(len(korean_symbol))

    symbol_to_id = {s: i for i, s in enumerate(korean_symbol)}

    text = '안녕하세요 3 분반'

    from jamo import hangul_to_jamo
    h2j = "".join(hangul_to_jamo(text))

    print([symbol_to_id[jamo] for jamo in h2j])
    print([jamo for jamo in h2j])










