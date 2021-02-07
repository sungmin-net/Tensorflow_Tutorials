# 210119_https://www.tensorflow.org/tutorials/load_data/unicode?hl=ko

import tensorflow as tf

print(tf.constant(u"Thanks 😊")) # print() added

print(tf.constant([u"You're", u"welcome!"]).shape) # print() added

# UTF-8로 인코딩된 string 스칼라로 표현한 유니코드 문자열
text_utf8 = tf.constant(u"语言处理")
print(text_utf8) # print() added

# UTF-16-BE 로 인코딩된 string 스칼라 표현 유니코드 문자열
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print(text_utf16be) # print() added

# 유니코드 코드 포인트의 벡터로 표현한 유니코드 문자열
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print(text_chars) # print() added

# 인코딩된 string 스칼라를 코드 포인트 벡터로 변환
print(tf.strings.unicode_decode(text_utf8, input_encoding = 'UTF-8')) # print() added

# 코드 포인트의 벡터를 인코드된 string 스칼라로 변환
print(tf.strings.unicode_encode(text_chars, output_encoding="UTF-8")) # print() added

# 인코드된 string 스칼라를 다른 인코딩으로 변환
print(tf.strings.unicode_transcode(text_utf8, input_encoding='UTF8', output_encoding='UTF-16-BE'))

# 배치 차원
# UTF-8 인코딩된 문자열로 표현한 유니코드 문자열의 배치
batch_utf8 = [s.encode('UTF-8') for s in 
              [u'hÃllo',  u'What is the weather tomorrow',  u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding = 'UTF-8')
for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)

batch_chars_padded = batch_chars_ragged.to_tensor(default_value = -1)
print(batch_chars_padded.numpy())

batch_chars_sparse = batch_chars_ragged.to_sparse()
print(tf.strings.unicode_encode(
        [[99, 97, 116], [100, 111, 103], [99, 111, 119]], output_encoding='UTF-8')) # print() added

print(tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8')) # print() added

# 패딩된 텐서나 희소(sparse)텐서는 unicode_encode 를 호출하기 전에 tf.RaggedTensor 로 변경
print(tf.strings.unicode_encode(tf.RaggedTensor.from_sparse(batch_chars_sparse),
        output_encoding = 'UTF-8')) # print() added

print(tf.strings.unicode_encode(tf.RaggedTensor.from_tensor(batch_chars_padded, padding = -1),
        output_encoding = 'UTF-8')) # print() added

# 유니코드 연산
# UTF-8에서 마지막 문자는 4바이트를 차지
# UTF8에서 마지막 문자는 4바이트를 차지합니다.
thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} 바이트; {}개의 UTF-8 문자'.format(num_bytes, num_chars))

# 부분 문자열
print(tf.strings.substr(thanks, pos = 7, len = 1).numpy())

# unit = 'UTF8_CHAR'로 지정하면 4바이트인 문자열 하나를 반환
print(tf.strings.substr(thanks, pos = 7, len = 1, unit = 'UTF8_CHAR').numpy())

# 유니코드 문자열 분리
print(tf.strings.unicode_split(thanks, 'UTF-8').numpy())

# 문자 바이트 오프셋
codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", 'UTF-8')
for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()):
    print("바이트 오프셋 {}: 코드 포인트 {}".format(offset, codepoint))
    
# 유니코드 스크립트 - 특정 char 가 어느 언어에 속하는 지를 구함
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']
print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]

print(tf.strings.unicode_script(batch_chars_ragged))

# 예제 - 간단한 분할
# dtype: string; shape: [num_sentences]
#
# 처리할 문장들 입니다. 이 라인을 수정해서 다른 입력값을 시도해 보세요!
sentence_texts = [u'Hello, world.', u'世界こんにちは']

sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
print(sentence_char_codepoint)

sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)

# 스크립트 식별자를 사용하여 단어 경계가 추가될 위치를 결정
sentence_char_starts_word = tf.concat([tf.fill([sentence_char_script.nrows(), 1], True),
        tf.not_equal(sentence_char_script[:, 1:],
        sentence_char_script[:, :-1])], axis = 1)

# word_starts[i]는 (모든 문장의 문자를 일렬로 펼친 리스트에서) i번째 단어가 시작되는 문자의 인덱스
word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis = 1)
print(word_starts)

# 시작 오프셋을 사용하여 전체 배치에 있는 단어 리스트를 담은 RaggedTensor 를 생성
word_char_codepoint = tf.RaggedTensor.from_row_starts(values = sentence_char_codepoint.values,
        row_starts = word_starts)
print(word_char_codepoint)

sentence_num_words = tf.reduce_sum(tf.cast(sentence_char_starts_word, tf.int64), axis = 1)
# sentence_word_char_codepoint[i, j, k]는 i번째 문장 안에 있는 j 번째 단어 안의 k 번째 문자의 코드포인트
sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(values = word_char_codepoint,
        row_lengths = sentence_num_words)
print(sentence_word_char_codepoint) 

# 최종 결과를 읽기 쉽게 utf-8 문자열로 다시 인코딩
print(tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list()) # print() added