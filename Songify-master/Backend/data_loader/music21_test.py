from music21 import converter, note, chord, stream, meter, key

filepath = 'data\\a-a-day-to-remember-downfall-of-us-all-intro-and-verse_key.mid'
midi = converter.parse(filepath)
midi.show('text')
# TimeSig = midi.recurse().getElementsByClass(meter.TimeSignature)[0]
# print(TimeSig.numerator, TimeSig.denominator) # get time signature 得到拍号
# KeySig = midi.recurse().getElementsByClass(key.KeySignature)[0]
# print(KeySig, KeySig.sharps) # get key signature 得到调号
for n in midi.recurse().notes:
    n.transpose(1, inPlace=True) # try transposing 尝试移调
print('------------------')
midi.show('text')
