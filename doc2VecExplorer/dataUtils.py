import os


proj_root_dir = """C:\Users\jkerxhalli\Desktop\golf\doc2vec\\"""
files_dir = "TestAndTrainDocs"
data_files_dir = proj_root_dir + files_dir + "\\"


def check_data_files(filer_dir = ""):
    file_list = []
    for root, subdirs, files in os.walk(data_files_dir+filer_dir):
        [file_list.append(root.replace(data_files_dir, "") + "\\" + f) if root != data_files_dir else file_list.append(f) for f in files]
    return file_list


def concatenate_files(label, file_dir):
    files = check_data_files(file_dir)
    concat_file_path = os.path.join(data_files_dir + file_dir + "\\", label + ".cor")
    concat_file = open(concat_file_path, "a+")
    for f in files:
        lines = open(data_files_dir + f).readlines()
        [concat_file.write(line) for line in lines]
    return label


def get_corpus_file(corpus_name):
    corpus_file = data_files_dir + corpus_name
    lines = open(corpus_file).readlines()
    return [line for line in lines]
