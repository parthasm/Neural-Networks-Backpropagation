file_n = "breast-cancer-wisconsin.data"
file_write_obj = open("breast_cancer_data_modified.csv",'w')
for line in open(file_n):
    if line.find('?')!=-1:
        continue
    file_write_obj.write(line[line.index(',')+1:])

file_write_obj.close()


