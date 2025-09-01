import numpy as np
import os
import shutil



# to split test data into vali
base_path = "/speech/nishanth/clean_research/gd_1.01_1.03_1.06_ptdb"
noise = ['0', '5', '10', '20', 'clean']

for i in noise:
    noise_path = os.path.join(base_path, i)
    train_path = os.path.join(noise_path, 'train')
    
    files = os.listdir(train_path)

    test_files = files[:600]
    valid_files = files[600:]

    valid_path = os.path.join(noise_path, 'valid')
    os.makedirs(valid_path, exist_ok = True)

    for file in valid_files:
        src = os.path.join(train_path, file)
        dst = os.path.join(valid_path, file)
        os.system(f'mv "{src}" "{dst}"')
'''

# to make a new file and move the final data as train test and valid
path = "/speech/nishant/roots_exps/data/"
dst_path = '/speech/nishant/clean_research/residual_exps/all_final_data'
base_path = '/speech/nishant/clean_research/residual_exps/all_residual'
noises = ['0', '5', '10', '20', 'clean']
splits = ['train', 'valid', 'test']
gender_dict = {"M":"MALE", "F":"FEMALE"}
for i in noises:
    for j in splits:
        if splits != 'test':
            dst_path1 = os.path.join(dst_path, j, "residual")
            os.makedirs(dst_path1, exist_ok = True)
            file_path = os.path.join(base_path, i ,j)
            files = os.listdir(file_path)
            for file in files:
                src = os.path.join(file_path, file)
                dst = os.path.join(dst_path1, file)
                shutil.copy(src, dst)
        else:
            dst_path1 = os.path.join(dst_path, j, i,"residual")
            os.makedirs(dst_path1, exist_ok = True)
            file_path = os.path.join(base_path, i ,j)
            files = os.listdir(file_path)
            for file in files:
                src = os.path.join(file_path, file)
                dst = os.path.join(dst_path1, file)
                shutil.copy(src, dst)

#os.makedirs(out_path)
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




for s in splits:
    if s != "test":
        delays = os.listdir(f"/speech/nishant/clean_research/residual_exps/all_final_data/{s}/residual")
        out_path = f"/speech/nishant/clean_research/residual_exps/all_final_data/{s}/labels"
        for file in delays:
            get_label(file)
    else:
        for n in noises:
            delays = os.listdir(f"/speech/nishant/clean_research/residual_exps/all_final_data/{s}/{n}/residual")
            out_path = f"/speech/nishant/clean_research/residual_exps/all_final_data/{s}/{n}/labels"
            for file in delays:
                get_label(file)



for s in splits:
    if s != "test":
        label_files = os.listdir(os.path.join(dst_path, s ,"labels"))
        for file_name in label_files:
            gd_arr = np.load(os.path.join(dst_path, s , "residual", file_name))
            labels = np.load(os.path.join(dst_path, s,  "labels", file_name))

            print(gd_arr.shape, labels.shape)

            len_diff = gd_arr.shape[1] - labels.shape[0]
            if gd_arr.shape[0] > labels.shape[0]:
                new_gd_arr = gd_arr[:labels.shape[0], :]
                np.save(os.path.join(dst_path, s ,"residual", file_name), new_gd_arr)
            else:
                new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[0], 255))), axis=1)
                np.save(os.path.join(dst_path,s, "residual", file_name), new_gd_arr)

    else:
        for q in noises:
            label_files = os.listdir(os.path.join(dst_path, s ,q ,"labels"))
            for file_name in label_files:
                gd_arr = np.load(os.path.join(dst_path, s ,q, "residual", file_name))
                labels = np.load(os.path.join(dst_path, s,q,  "labels", file_name))

                print(gd_arr.shape, labels.shape)

                len_diff = gd_arr.shape[1] - labels.shape[0]
                if gd_arr.shape[0] > labels.shape[0]:
                    new_gd_arr = gd_arr[:labels.shape[0], :]
                    np.save(os.path.join(dst_path, s ,q,"residual", file_name), new_gd_arr)
                else:
                    new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[0], 255))), axis=1)
                    np.save(os.path.join(dst_path,s, q, "residual", file_name), new_gd_arr)                            

'''
'''dst_path = '/speech/nishant/clean_research/final_data/test/group_delay'
src_path = '/speech/nishant/clean_research/final_data/test'
files = os.listdir(src_path)
for file in files:
    src = os.path.join(src_path, file)
    dst = os.path.join(dst_path, file)
    shutil.move(src, dst)
'''
