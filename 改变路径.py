import os
import re

filename = "/home/lqb/project/SSL_Anti-spoofing-main/data-dialect/test/utt2spk"
to_path = "/home/lqb/project/SSL_Anti-spoofing-main/Dataset-AP20-dialect/IEMOCAP/labels/test_utt2spk.txt"


def write(path, contxt):
    with open(path, "a+", encoding='utf-8') as f:
        f.write(contxt)
        f.close()


with open(filename, 'r') as raw:
    # print(raw)

    for line in raw:
        print(line)
        x1 = line.split(" ")[0]
        x2 = line.split(" ")[1]
        # x2 = line.split(" ")[1].replace('\n','')
        # print(x2)
        # y1 = x2.split("/")[0]
        # print(y1)
        # y2 = x2.split("/")[-1]
        # print(x)
        # print("F:/ADI17/fbank/dev_shuffle/{}".format(x))

        contxt = "- " + x1 + " - - " + x2
        print(contxt)
        write(to_path, contxt)

