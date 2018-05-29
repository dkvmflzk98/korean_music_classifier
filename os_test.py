import os

for root, dirs, files in os.walk('./music/balad'):
    for fname in files:
        full_fname = os.path.join(root, fname)

        print(full_fname)