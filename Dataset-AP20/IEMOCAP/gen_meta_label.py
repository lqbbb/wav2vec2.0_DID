import os
import numpy as np
import json
import random
from pathlib import Path
import re
import sys

# IEMOCAP_DIR = Path(sys.argv[1])



print ('Generating metalabels...')
metalabel = {}

    # for labelfile in label_dir.rglob('*.txt'):

labelfile = "/home/lqb/PycharmProjects/FT-w2v2-test-AP20/test_utt2spk"
with open(labelfile, 'r') as f:
    for full_audio_name in f:
        audio_name = full_audio_name.split(" ")[0]
        label_name = full_audio_name.split(" ")[1].replace('\n','')
            # m = re.match(r".*(Ses.*)\t(.*)\t.*", line)
            #     if m:
            #         name, label = m.groups()
        metalabel[audio_name+'.wav'] = label_name
with open(f'metalabel.json', 'w') as f:
    json.dump(metalabel, f, indent=4)
