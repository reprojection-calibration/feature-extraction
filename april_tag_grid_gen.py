import numpy as np

tag = np.full((6, 6), "d", dtype='U1')

tag = np.pad(tag, (1), 'constant', constant_values=("w"))
tag = np.pad(tag, (1), 'constant', constant_values=("b"))

black_white_tag = np.pad(tag, (1), 'constant', constant_values=("w"))
black_white_tag = np.pad(black_white_tag, (1), 'constant', constant_values=("w"))

black_white_tag[0 + 1][0 + 1] = "b"
black_white_tag[0 + 1][0] = "b"
black_white_tag[0][0 + 1] = "b"

black_white_tag[0 + 1][-1 - 1] = "b"
black_white_tag[0][-1 - 1] = "b"
black_white_tag[0 + 1][-1] = "b"

black_white_tag[-1 - 1][0 + 1] = "b"
black_white_tag[-1][0 + 1] = "b"
black_white_tag[-1 - 1][0] = "b"

black_white_tag[-1 - 1][-1 - 1] = "b"
black_white_tag[-1][-1 - 1] = "b"
black_white_tag[-1 - 1][-1] = "b"

print(black_white_tag)

black_white_code = ["".join(item) for item in black_white_tag.astype(str)]
black_white_code = "".join(black_white_code)
print(black_white_code)
