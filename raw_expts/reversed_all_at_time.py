import numpy as np
import os
import shutil



path = "/speech/nishanth/root_exps/data/"
dst_path = '/speech/nishanth/raw_exps/full_final_data_8khz_rev'
base_path = '/speech/nishanth/raw_exps/full_frames_8khz_rev'
noises = ['0', '5', '10', '20', 'clean']
splits = ['train', 'valid']
gender_dict = {"M":"MALE", "F":"FEMALE"}
for i in noises:
    for j in splits:
        if j != 'test':
            dst_path1 = os.path.join(dst_path, j, "raw")
            os.makedirs(dst_path1, exist_ok = True)
            file_path = os.path.join(base_path, i ,j)
            files = os.listdir(file_path)
            for file in files:
                new_file = i+"r"+"_"+file
                src = os.path.join(file_path, file)
                dst = os.path.join(dst_path1, new_file)
                shutil.copy(src, dst)
        else:
            dst_path2 = os.path.join(dst_path, j, i,"raw")
            os.makedirs(dst_path2, exist_ok = True)
            file_path = os.path.join(base_path, i ,j)
            files = os.listdir(file_path)
            for file in files:
                new_file = i+"r"+"_"+file
                src = os.path.join(file_path, file)
                dst = os.path.join(dst_path2, new_file)
                shutil.copy(src, dst)

def get_pitch(link, outname):
    data = open(link, "rt").readlines()
    data = [round(float(x.strip().split(maxsplit=3)[0]), 3) for x in data]
    data = data[::-1]

    np.save(os.path.join(out_path, outname), np.array(data))

def get_label(file):
    noise, _, gender, id = file.split("_")

    filename = "ref_" + gender + "_" + id.split(".")[0] + ".f0"
    # out_name = "mic_" + gender + "_" + id.split(".")[0]
    out_name = file.split(".")[0]
    print(out_name)

    sub_path = os.path.join(path, gender_dict[gender[0]], "REF", gender, filename)
    get_pitch(sub_path, out_name)

######GET LABELS #######
for s in splits:
    if s != "test":
        delays = os.listdir(f"{dst_path}/{s}/raw")
        out_path = f"{dst_path}/{s}/labels"
        os.makedirs(out_path, exist_ok = True)
        for file in delays:
            get_label(file)
    else:
        for n in noises:
            delays = os.listdir(f"{dst_path}/{s}/{n}/raw")
            out_path = f"{dst_path}/{s}/{n}/labels"
            os.makedirs(out_path, exist_ok = True)
            for file in delays:
                get_label(file)


######## FIX LENGTH #######
for s in splits:
    if s != "test":
        label_files = os.listdir(os.path.join(dst_path, s ,"labels"))
        for file_name in label_files:
            gd_arr = np.load(os.path.join(dst_path, s , "raw", file_name))
            labels = np.load(os.path.join(dst_path, s,  "labels", file_name))
            
            print(gd_arr.shape, labels.shape)

            len_diff = gd_arr.shape[1] - labels.shape[0]
            if gd_arr.shape[0] >= labels.shape[0]:
                new_gd_arr = gd_arr[:labels.shape[0], :]
            else:
                new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[0], 255))), axis=1)
                
            new_file_name = file_name

                # Save the new raw file
            np.save(os.path.join(dst_path, s, "raw", new_file_name), new_gd_arr)

                # Save the new labels file with the same prefix
            np.save(os.path.join(dst_path, s,"labels", new_file_name), labels)
    else:
        for q in noises:
            label_files = os.listdir(os.path.join(dst_path, s ,q ,"labels"))
            for file_name in label_files:
                gd_arr = np.load(os.path.join(dst_path, s ,q, "raw", file_name))
                labels = np.load(os.path.join(dst_path, s,q,  "labels", file_name))
                
                print(gd_arr.shape, labels.shape)

                len_diff = gd_arr.shape[1] - labels.shape[0]
                if gd_arr.shape[0] >= labels.shape[0]:
                    new_gd_arr = gd_arr[:labels.shape[0], :]
                    
                else:
                    new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[0], 255))), axis=1)
                                               
                
                
                new_file_name =  file_name

                # Save the new raw file
                np.save(os.path.join(dst_path, s, q, "raw", new_file_name), new_gd_arr)

                # Save the new labels file with the same prefix
                np.save(os.path.join(dst_path, s, q, "labels", new_file_name), labels)