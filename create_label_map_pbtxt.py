import sys
sys.path.append("../API")

from alet import ALET

dataset = ALET("../data/ALET/new_train","../data/ALET/new_train.json")      

with open('alet_label_map.pbtxt', 'a+') as the_file:
    for item in dataset.categories:
        the_file.write('item\n')
        the_file.write('{\n')
        the_file.write('id :{}'.format(item['id']))
        the_file.write('\n')
        the_file.write("name :'{0}'".format(item['name']))
        the_file.write('\n')
        the_file.write('}\n')
