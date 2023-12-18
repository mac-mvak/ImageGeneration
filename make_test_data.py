import shutil
import os

os.makedirs('gen_data/data', exist_ok=True)
os.makedirs('test_data/data', exist_ok=True)

for i in range(1, 2001):
    shutil.copy(f'data/data/{i}.png', f'test_data/data/{i}.png')


