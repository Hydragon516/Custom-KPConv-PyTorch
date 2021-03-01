import os
import numpy as np
import math
import random

# dirname = "/HDD1/mvpservereight/minhyeok/PointCloud/object_float_v5"

# cnt = 0

# for (path, dir, files) in os.walk(dirname):
#     for filename in files:
#         ext = os.path.splitext(filename)[-1]
#         if ext == '.bin':
#             cnt += 1

#             A = np.fromfile(os.path.join(path, filename), dtype='float32', sep="")
#             n = A.shape[0] // 4
#             A = A.reshape([n, 4])
#             A[:, 3] = 1

#             CM = np.average(A[:,:3], axis=0, weights=A[:,3])
#             xyz = A[:, 0:3]

#             shift_A = xyz - CM

#             max_r = 0

#             for i in range(n):
#                 x = shift_A[i][0]
#                 y = shift_A[i][1]
#                 z = shift_A[i][2]

#                 r = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))

#                 if r > max_r:
#                     max_r = r
            
#             norm_shift_A = shift_A / max_r

#             new_path = "/".join((path.replace("object_float_v5", "object_float_v5_txt").split("/"))[:-1])
#             os.makedirs(new_path, exist_ok=True)

#             new_f = open(new_path + "/" + str(cnt) + ".txt", 'w')

#             print(new_path + "/" + str(cnt) + ".txt")

#             for i in range(len(norm_shift_A)):
#                 new_f.write(str(norm_shift_A[i][0]) + "," + str(norm_shift_A[i][1]) + "," + str(norm_shift_A[i][2]) \
#                     + ",0,0,0\n")

#             new_f.close()

dirname = "/HDD1/mvpservereight/minhyeok/PointCloud/object_float_v5_txt"

lists = []

for (path, dir, files) in os.walk(dirname):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            lists.append(path + "/" + filename + "\n")

random.shuffle(lists)

f = open(os.path.join(dirname, "train.txt"), 'w')
            
for i in range(len(lists)):
    f.write(lists[i])

f.close()