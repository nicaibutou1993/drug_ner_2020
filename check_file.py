import os
import zipfile


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def read_txt_ann(ann_path):
    fp = open(ann_path, encoding='utf-8')
    for line in fp:
        line = line.strip()
        # print(ann_path,line)
        if line != '':
            # if len(line.split('\t')) == 1:
            # print('-' * 10, ann_path, line)
            _, temp_data, words = line.split('\t')
            name, start, end = temp_data.split(' ')
            start, end = int(start), int(end)

            try:
                assert (end - start) == len(words)
            except Exception as e:
                print(e)
            # if (end - start) != len(words):
            # print('_' * 10, ann_path, line)


if __name__ == '__main__':
    zip_src = 'G:/tmp/test_an.zip'
    dst_dir = 'G:/tmp/output'
    unzip_file(zip_src, dst_dir)
    for i in range(1000, 1500):
        txt_path = dst_dir + '/{}.ann'.format(i)
        # if os.path.exists(txt_path):
        read_txt_ann(txt_path)
