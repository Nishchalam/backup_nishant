import os
import numpy as np

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


path = "/speech/nishant/roots_exps/data/"
gender_dict = {"M":"MALE", "F":"FEMALE"}



out_path = "/speech/nishant/clean_research/residual_exps/all_final_data/train/labels"
delays = os.listdir("/speech/nishant/clean_research/residual_exps/all_final_data/train/residual")
#os.makedirs(out_path)

for file in delays:
    get_label(file)



