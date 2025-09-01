import numpy as np
import os
import shutil



path = "/speech/nishanth/roots_exps/data"
dst_path = '/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit_final'
base_path = '/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit'
noises = ['0', '5', '10', '20', 'clean']
splits = ['train', 'valid', 'test']
gender_dict = {"M":"MALE", "F":"FEMALE"}
for i in noises:
    for j in splits:
        if j != 'test':
            dst_path1 = os.path.join(dst_path, j, "gd")
            os.makedirs(dst_path1, exist_ok = True)
            file_path = os.path.join(base_path, i ,j)
            files = os.listdir(file_path)
            for file in files:
                new_file= i+"_"+ file
                src = os.path.join(file_path, file)
                dst = os.path.join(dst_path1, new_file)
                shutil.copy(src, dst)
        else:
            dst_path2 = os.path.join(dst_path, j, i,"gd")
            os.makedirs(dst_path2, exist_ok = True)
            file_path = os.path.join(base_path, i ,j)
            files = os.listdir(file_path)
            for file in files:
                new_file= i+"_"+ file
                src = os.path.join(file_path, file)
                dst = os.path.join(dst_path2, new_file)
                shutil.copy(src, dst)

def get_pitch(link, outname):
    data = open(link, "rt").readlines()
    data = [round(float(x.strip().split(maxsplit=3)[0]), 3) for x in data]

    np.save(os.path.join(out_path, outname), np.array(data))

def get_label(file):
    noise,_, gender, id = file.split("_")

    filename = "ref_" + gender + "_" + id.split(".")[0] + ".f0"
    out_name = noise + "_"+"mic_" + gender + "_" + id.split(".")[0]
    print(out_name)

    sub_path = os.path.join(path, gender_dict[gender[0]], "REF", gender, filename)
    get_pitch(sub_path, out_name)


######GET LABELS #######
for s in splits:
    if s != 'test':
        delays = os.listdir(f"{dst_path}/{s}/gd")

        out_path = f"{dst_path}/{s}/labels"
        os.makedirs(out_path, exist_ok = True)
        for file in delays:
            res = np.load(os.path.join(f"{dst_path}/{s}/gd", file))
            #res = res[:,:,:]
            np.save(os.path.join(f"{dst_path}/{s}/gd", file), res)
            get_label(file)
    else:
        for n in noises:
            delays = os.listdir(f"{dst_path}/{s}/{n}/gd")

            out_path = f"{dst_path}/{s}/{n}/labels"
            os.makedirs(out_path, exist_ok = True)
            for file in delays:
                res = np.load(os.path.join(f"{dst_path}/{s}/{n}/gd", file))
                #res = res[:,:,:]
                np.save(os.path.join(f"{dst_path}/{s}/{n}/gd", file), res)
                get_label(file)


######## FIX LENGTH #######
for s in splits:
    if s != 'test':
        label_files = os.listdir(os.path.join(dst_path, s ,"labels"))
        for file_name in label_files:
            gd_arr = np.load(os.path.join(dst_path, s , "gd", file_name))
            labels = np.load(os.path.join(dst_path, s,  "labels", file_name))
            labels = np.round(labels)

            print(gd_arr.shape, labels.shape)

            len_diff = gd_arr.shape[1] - labels.shape[0]
            if gd_arr.shape[1] > labels.shape[0]:
                new_gd_arr = gd_arr[:,:labels.shape[0], :]
                np.save(os.path.join(dst_path, s ,"gd", file_name), new_gd_arr)
            else:
                new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[1], 511))), axis=1)
                np.save(os.path.join(dst_path,s, "gd", file_name), new_gd_arr)

    else:
        for q in noises:
            label_files = os.listdir(os.path.join(dst_path, s ,q ,"labels"))
            for file_name in label_files:
                gd_arr = np.load(os.path.join(dst_path, s ,q, "gd", file_name))
                labels = np.load(os.path.join(dst_path, s,q,  "labels", file_name))
                labels = np.round(labels)
                print(gd_arr.shape, labels.shape)

                len_diff = gd_arr.shape[1] - labels.shape[0]
                if gd_arr.shape[1] > labels.shape[0]:
                    new_gd_arr = gd_arr[:,:labels.shape[0], :]
                    np.save(os.path.join(dst_path, s ,q,"gd", file_name), new_gd_arr)
                else:
                    new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[0], 511))), axis=1)
                    np.save(os.path.join(dst_path,s, q, "gd", file_name), new_gd_arr)                            
