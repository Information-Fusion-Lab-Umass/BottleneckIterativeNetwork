import os.path
import urllib.request
from tqdm import tqdm


categories = ["Babble", "Cafe", "Car", "LR", "Street", "White"]
db_level = ["-5", "0", "5", "10", "15", "20"]

noise_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_NOISE"

for cat in tqdm(categories):
    for db in db_level:
        dl_link = f"https://zenodo.org/records/1172064/files/{cat}_{db}.tar.gz?download=1"
        urllib.request.urlretrieve(dl_link, filename=os.path.join(noise_path, f"{cat}_{db}.tar.gz"))