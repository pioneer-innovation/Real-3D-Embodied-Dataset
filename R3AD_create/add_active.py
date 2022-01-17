active_data_path = '../data/R3AD_trainval/add.txt'
rand_data_path = '../data/R3AD_trainval/random.txt'

cloud_path = '../data/R3AD_trainval/cloud_path.txt'
cloud2_path = '../data/R3AD_trainval/cloud_path_副本.txt'

anno_path = '../data/R3AD_trainval/anno_path.txt'
anno2_path = '../data/R3AD_trainval/anno_path_副本.txt'

train_id_path = '../data/R3AD_trainval/train_data_idx.txt'
train_id_2_path = '../data/R3AD_trainval/train_data_idx_副本.txt'

active_cloud_path = '../data/active_R3AD_trainval/cloud_path.txt'
active_anno_path = '../data/active_R3AD_trainval/anno_path.txt'

active_data_list = []
active_anno_list = []
rand_data_list = []

data_list = []
anno_list = []

id_list=[]
active_id_list=[]

# active_id_list=[str(i) for i in range(4664,4864)]

cloud_name_list = ['cloud 0.pcd','cloud 45.pcd','cloud 90.pcd','cloud 135.pcd','cloud 180.pcd','cloud 225.pcd','cloud 270.pcd','cloud 315.pcd']
anno_name_list = ['anno 0.txt','anno 45.txt','anno 90.txt','anno 135.txt','anno 180.txt','anno 225.txt','anno 270.txt','anno 315.txt']

f = open(active_data_path, "r")
lines = f.readlines()  # 读取全部内容
for line in lines:
    line = line.replace('\n', '').replace('./data/', '')
    active_data_list.append(line)
f.close()

for i in range(len(active_data_list)):
    tmp = active_data_list[i].split('/')
    ind = cloud_name_list.index(tmp[2])
    tmp2 = anno_name_list[ind]
    tmp[2] = tmp2
    tmp = '/'.join(tmp)
    active_anno_list.append(tmp)

# f = open(rand_data_path, "r")
# lines = f.readlines()  # 读取全部内容
# for line in lines:
#     line = line.replace('\n', '').replace('./data/', '')
#     rand_data_list.append(line)
# f.close()

f = open(cloud2_path, "r")
lines = f.readlines()  # 读取全部内容
for line in lines:
    line = line.replace('\n', '')
    data_list.append(line)
f.close()

f = open(anno2_path, "r")
lines = f.readlines()  # 读取全部内容
for line in lines:
    line = line.replace('\n', '')
    anno_list.append(line)
f.close()

f = open(train_id_2_path, "r")
lines = f.readlines()  # 读取全部内容
for line in lines:
    line = line.replace('\n', '')
    id_list.append(line)
f.close()

for i in range(len(active_data_list)):
    data_list.append(active_data_list[i])

for i in range(len(active_anno_list)):
    anno_list.append(active_anno_list[i])

for i in range(len(active_id_list)):
    id_list.append(active_id_list[i])

f = open(train_id_path, "w")
for id in id_list:
    f.write(id + '\n')
f.close()

f = open(cloud_path, "w")
for id in data_list:
    f.write(id + '\n')
f.close()

f = open(anno_path, "w")
for id in anno_list:
    f.write(id + '\n')
f.close()