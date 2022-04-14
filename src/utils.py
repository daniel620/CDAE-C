import os


def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs

        
def files_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(files):
            print("sub_files:", files)
            return files
        
def save_file2csvv2(file_dir, file_name,label):
    """
    save file path to csv,this is for classification
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :param label:classification label
    :return:
    """
    out = open(file_name, 'w')
    sub_files = files_name_path(file_dir)
    out.writelines("class,filename" + "\n")
    for index in range(len(sub_files)):
        out.writelines(label+","+file_dir + "/" + sub_files[index] + "\n")
       

def save_file2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    sub_dirs = file_name_path(file_dir)
    out.writelines("filename" + "\n")
    for index in range(len(sub_dirs)):
        out.writelines(file_dir + "/" + sub_dirs[index] + "\n")

files_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/3-noisy_bmp'
csv_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/3-noisy_bmp.csv'
save_file2csv(files_path, csv_path)

# save_file2csv("/Volumes/Secrets/Research_DAE/LUNA16-Lung-Nodule-Analysis-2016-Challenge-master/LUNA16Challege/data/segmentation/Image", "/Volumes/Secrets/Research_DAE/LUNA16-Lung-Nodule-Analysis-2016-Challenge-master/LUNA16Challege/data/segmentation/Image.csv")
