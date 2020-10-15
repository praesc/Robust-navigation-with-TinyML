# =============================================================================
# Author : Miguel de Prado
# Comments: Create a Morse dataset
# =============================================================================
from PIL import Image
import numpy as np
import random
import os

# Point is a color pixel
# Line is three color pixel
# The space between parts of the same letter is one white pixel
point = [1, 1, 0, 0]
line = [1, 1, 1, 1, 1, 1, 0, 0]

A = point + line
B = line + point + point + point
C = line + point + line + point
O = line + line + line
P = point + line + line + point
Q = line + line + point + line
SOS = point + point + point + line + line + line + point + point + point
words = [A, B, C, O, P, Q, SOS]

words_str = ['A', 'B', 'C', 'O', 'P', 'Q', 'SOS']

line_length = 143
for idx_w, word in enumerate(words):
    for idx in range(1000):
        line = np.ones(line_length, dtype=np.uint8) * 255

        # Sample phrase
        pos = random.randint(0, line_length -1 - len(SOS))
        intensity = random.randint(0, 128)
        sec_word = [intensity * value if value == 1 else 255 for value in word]
        line[pos:pos + len(word)] = sec_word

        # Convert and save
        im = Image.fromarray(line.reshape(1, 143))
        path = os.path.join('datasets', 'morse', 'training', words_str[idx_w], 'Image_' + str(idx) + '.jpeg')
        im.save(path)



