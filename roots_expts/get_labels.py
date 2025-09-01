import os
import numpy as np

path = "/speech/nishanth/roots_exps/data/"
out_path = "/speech/nishanth/roots_exps/final_data/test/labels"

gender_dict = {"M":"MALE", "F":"FEMALE"}
delays = os.listdir("/speech/nishanth/roots_exps/final_data/test/group_delay")

def get_pitch(link, outname):
    data = open(link, "rt").readlines()
    data = [round(float(x.strip().split(maxsplit=3)[0]), 3) for x in data]

    np.save(os.path.join(out_path, outname), np.array(data))

def get_label(file):
    _, gender, id = file.split("_")

    filename = "ref_" + gender + "_" + id.split(".")[0] + ".f0"
    out_name = "mic_" + gender + "_" + id.split(".")[0]
    print(out_name)

    sub_path = os.path.join(path, gender_dict[gender[0]], "REF", gender, filename)
    get_pitch(sub_path, out_name)


for file in delays:
    get_label(file)



