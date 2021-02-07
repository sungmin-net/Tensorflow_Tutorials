# 210119_https://www.tensorflow.org/tutorials/tensorflow_text/intro?hl=ko

import tensorflow as tf
import tensorflow_text as text

# 유니코드
docs = tf.constant([u'Everything not saved will be lost.'.encode('UTF-16-BE'),
        u'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')

# 토큰화
# whitespace
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())

# unicodeScript
tokenizer = text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())

# unicode
tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
print(tokens.to_list())

# Offset
tokenizer = text.UnicodeScriptTokenizer()
(tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
print(offset_starts.to_list())
print(offset_limits.to_list())

# TF.Data
docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["it's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x : tokenizer.tokenize(x))
iterator = iter(tokenized_docs)
print(next(iterator).to_list())
print(next(iterator).to_list())

# 워드 셰이프
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])

# Is capitalized?
f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
# Are all letters uppercased?
f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
# Does the token contain punctuation?
f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
# Is the token a number?
f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

print(f1.to_list())
print(f2.to_list())
print(f3.to_list())
print(f4.to_list())

# N-gram sliding window
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])

# Ngrams, in this case bi-gram (n = 2)
bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(bigrams.to_list())
