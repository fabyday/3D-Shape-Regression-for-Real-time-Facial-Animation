from tqdm import tqdm, trange
from time import sleep


for i in tqdm(range(100)):
    # Print using tqdm class method .write()
    sleep(0.1)
    if not (i % 3):
        tqdm.write("Done task %i" % i)
    # Can also use bar.write()