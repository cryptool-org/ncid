"""Framework-free cipher type constants.

This module is part of the inference code path and therefore must not import
any training framework (or anything that does). `cipherTypeDetection.config`
re-exports these names so that `config.CIPHER_TYPES` etc. keep working.
"""

ROTOR_CIPHER_TYPES = ['enigma', 'm209', 'purple', 'sigaba', 'typex']
# CIPHER_TYPES = ['columnar_transposition', 'hill', 'playfair', 'simple_substitution', 'vigenere']
CIPHER_TYPES = ['amsco', 'autokey', 'baconian', 'bazeries', 'beaufort', 'bifid', 'cadenus', 'checkerboard', 'columnar_transposition',
                'condi', 'cmbifid', 'digrafid', 'foursquare', 'fractionated_morse', 'grandpre', 'grille', 'gromark', 'gronsfeld',
                'headlines', 'homophonic', 'key_phrase',  # , 'incomplete_columnar_transposition'
                'monome_dinome', 'morbit', 'myszkowski', 'nicodemus', 'nihilist_substitution', 'nihilist_transposition', 'null',
                'numbered_key', 'periodic_gromark', 'phillips', 'phillips_rc', 'plaintext', 'playfair', 'pollux', 'porta', 'portax',
                'progressive_key', 'quagmire1', 'quagmire2', 'quagmire3', 'quagmire4', 'ragbaby', 'railfence', 'redefence',
                'route_transposition', 'running_key', 'seriated_playfair', 'slidefair', 'swagman', 'tridigital', 'trifid', 'tri_square',
                'two_square', 'variant', 'vigenere', 'enigma', 'm209', 'purple', 'sigaba', 'typex']

MTC3 = 'mtc3'
ACA = 'aca'
ROTOR = 'rotor'
ALL = "all"
