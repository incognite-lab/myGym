import numpy as np
import sys
import os

path = sys.argv[1]

if not os.path.exists(path):
    print(f"Not found: {path}")
    sys.exit(1)

# Road file
data = np.load(path, allow_pickle=True)

# information
print(f"\nfile path: {path}")
print(f"shape: {data.shape}")
print(f"dtype: {data.dtype}")

# ---- 中身の一部を表示 ----
np.set_printoptions(precision=4, suppress=True)  # 小数点をきれいに
print("\ndata")
print(data[:50])  