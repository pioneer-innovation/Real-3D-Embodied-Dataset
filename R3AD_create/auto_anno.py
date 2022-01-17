import numpy as np
import open3d as o3d
import os
import copy

def AngleLimit(angle):
    angle_l = angle.copy()
    if angle>3.1415926:
        angle_l = angle%(3.1415926*2)
        if angle_l > 3.1415926:
            angle_l-=3.1415926*2
    elif angle<-3.1415926:
        angle_l = -(-angle%(3.1415926*2))
        if angle_l < -3.1415926:
            angle_l+=3.1415926*2
    return angle_l

class DetectObject():
    def __init__(self,array,number,classes):
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]

        self.l = array[3]
        self.w = array[4]
        self.h = array[5]

        self.roll = array[6]
        self.coor_roll = 0

        self.num = number
        self.classes=classes
        self.array = array

    def AngleObjToCoor(self,coor):
        vec = [coor.x - self.x, coor.y-self.y, coor.z-self.z]
        alpha = np.arctan(vec[1]/vec[0])
        beta = np.arctan(vec[2]/np.hypot(vec[0],vec[1]))
        if (vec[0]<0 and vec[1]<0)or(vec[0]<0 and vec[1]>0):
            alpha = alpha - 3.1415
        return AngleLimit(alpha-self.roll), beta

    def rotate(self,corner,roll):
        rotate_matrix = np.array([np.cos(roll),-np.sin(roll),0,
                                  np.sin(roll),np.cos(roll),0,
                                  0           ,0           ,1]).reshape(3,3)
        return np.matmul(rotate_matrix,corner.T).T

    def corner(self):
        corner1=np.array([+ self.l / 2, + self.w / 2, + self.h / 2])
        corner2=np.array([- self.l / 2, + self.w / 2, + self.h / 2])
        corner3=np.array([+ self.l / 2, - self.w / 2, + self.h / 2])
        corner4=np.array([- self.l / 2, - self.w / 2, + self.h / 2])

        corner5=np.array([+ self.l / 2, + self.w / 2, - self.h / 2])
        corner6=np.array([- self.l / 2, + self.w / 2, - self.h / 2])
        corner7=np.array([+ self.l / 2, - self.w / 2, - self.h / 2])
        corner8=np.array([- self.l / 2, - self.w / 2, - self.h / 2])

        self.corners = np.concatenate((corner1,corner2,corner3,corner4,
                                       corner5,corner6,corner7,corner8),axis=0).reshape(-1,3)
        self.center = np.array([self.x,self.y,self.z])

        self.corners = self.rotate(self.corners,self.roll-self.coor_roll) # box 自转
        self.center = self.rotate(self.center,-self.coor_roll) # center 公转

        self.corners = self.corners+self.center

class Coor():
    def __init__(self,array):
        self.x = array[1]
        self.y = array[0]
        self.z = 1.1

        self.roll = array[2]
        self.max_alpha = 1.2 # 水平可见角度上限
        self.min_alpha = -1.2 # 水平可见角度下限
        self.max_beta = 0.78 # 竖直可见角度上限
        self.min_beta = -1.6 # 竖直可见角度下限

    def AngleCoorToObj(self,obj):
        vec = [obj.x - self.x, obj.y-self.y, obj.z-self.z]
        alpha = np.arctan(vec[1]/vec[0])
        beta = np.arctan(vec[2]/np.hypot(vec[0],vec[1]))
        if (vec[0]<0 and vec[1]<0):
            alpha = alpha - 3.1415
        elif (vec[0]<0 and vec[1]>=0):
            alpha = alpha + 3.1415
        if alpha>3.142:
            print("alpha error !!!!!!")
            print(alpha)
        alpha = alpha - self.roll
        return AngleLimit(alpha), beta

    def InAngleRange(self,obj_list):
        i=0
        while True:
            if i >=len(obj_list):
                break
            alpha_cto, beta_cto = self.AngleCoorToObj(obj_list[i])
            if alpha_cto<self.min_alpha or alpha_cto>self.max_alpha or beta_cto<self.min_beta or beta_cto>self.max_beta:
                obj_list.pop(i)
                i=i-1
            i=i+1
        return obj_list

    def AnnoObj(self,obj_list):
        obj_list = self.InAngleRange(obj_list)
        for i in range(0,len(obj_list)):
            obj_list[i].coor_roll = self.roll
            obj_list[i].x = obj_list[i].x - self.x
            obj_list[i].y = obj_list[i].y - self.y
        return obj_list

def AnnoToPc(corners):
    for i in range(0,len(corners)):
        corners[i] = corners[i][:,[1,2,0]]
        corners[i][:,2] = -corners[i][:,2]
        corners[i][:, 1] = corners[i][:, 1] - 1.1
    return corners

def DrawBox(corners):
    box_list=[]
    line_set = np.array([[0, 1], [1, 3], [0, 2], [2, 3],
                         [4, 5], [4, 6], [5, 7], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]])
    for i in range(0,len(corners)):
        corner_set = corners[i]
        color_set = np.array([[0, 1, 0] for j in range(len(line_set))])
        box = o3d.geometry.LineSet()
        box.points = o3d.utility.Vector3dVector(corner_set)
        box.lines = o3d.utility.Vector2iVector(line_set)
        box.colors = o3d.utility.Vector3dVector(color_set)
        box_list.append(box)
    return box_list

# 初始化路径
Home_1_path = './data/Home_1/'
coors = os.listdir(Home_1_path)
cloud_files = ['cloud 0','cloud 45','cloud 90','cloud 135','cloud 180','cloud 225','cloud 270','cloud 315']
anno_files = ['anno 0','anno 45','anno 90','anno 135','anno 180','anno 225','anno 270','anno 315']
label_path = 'label.txt'

# 读取label文件
f = open(label_path,"r")
lines = f.readlines()
f.close()
lines_num = len(lines)
for i in range(0,lines_num):
    lines[i] = lines[i].strip('\n')
    lines[i] = lines[i].split(' ')
lines = np.array(lines).astype(np.float)

# 生成原始obj列表
obj_list = []
for i in range(0,lines_num):
    obj = DetectObject(lines[i,1:8],lines[i,0],lines[i,-1])
    obj_list.append(obj)

# 手动校准
x_delta = 0
y_delta = 0
z_delta = 0
choose_id = -1
delta=0.01

# 按坐标进行操作
for (i,coor) in enumerate(coors):
    coor_path = Home_1_path + coor
    coor_f = np.array(coor.split(' ')).astype(np.float)*0.25
    coor_f = np.append(coor_f,0)
    if i < 136:
        continue
    for (j,file) in enumerate(cloud_files):
        if i <= 136 and j<6:
            continue
        # 初始化coor类
        coor_f[2] = j*0.785398
        coor_c = Coor(coor_f)
        save=0
        print(coor,file,'——',i,j)

        # 删去不在当前视角范围内的obj 更新obj公转角度（坐标轴自转角度）
        anno_obj_list = coor_c.AnnoObj(copy.deepcopy(obj_list))

        # 生成box corner list
        corners_list=[]
        for x in range(0,len(anno_obj_list)):
            anno_obj_list[x].corner()
            corners_list.append(anno_obj_list[x].corners)

        # 将正常坐标系转换为open3d坐标系
        corners_list = AnnoToPc(corners_list)

        # 生成box的边
        box_list = DrawBox(corners_list)
        # for x in range(0, len(box_list)):
        #     yzx = np.asarray(box_list[x].points)
        #     yzx[:, 0] += y_delta
        #     yzx[:, 1] += z_delta
        #     yzx[:, 2] += x_delta
        #     box_list[x].points = o3d.utility.Vector3dVector(yzx)

        # open3d显示说明
        #   1 Z   -X 2
        #     |  /
        #     | /
        #     0----> Y 0
        cloud_file_path = coor_path + '/' + file + '.pcd'
        pcd = o3d.io.read_point_cloud(cloud_file_path)
        # origin = o3d.geometry.PointCloud()
        # yzx = np.array([0,0,0])
        # origin.points = o3d.utility.Vector3dVector(yzx)

        def custom_draw_geometry_with_key_callback(pcd):
            def AddBox(vis):
                for x in range(0,len(box_list)):
                    vis.add_geometry(box_list[x])
                return False

            def Forward(vis):
                global x_delta,delta
                if choose_id>=0:
                    yzx = np.asarray(box_list[choose_id].points)
                    yzx[:, 2]-=delta
                    anno_obj_list[choose_id].center[0]+=delta
                    box_list[choose_id].points = o3d.utility.Vector3dVector(yzx)
                    vis.update_geometry(box_list[choose_id])
                else:
                    for x in range(0, len(box_list)):
                        yzx = np.asarray(box_list[x].points)
                        yzx[:, 2]-=delta
                        box_list[x].points = o3d.utility.Vector3dVector(yzx)
                        vis.update_geometry(box_list[x])
                    x_delta = x_delta - delta
                return False

            def Back(vis):
                global x_delta,delta
                if choose_id>=0:
                    yzx = np.asarray(box_list[choose_id].points)
                    yzx[:, 2]+=delta
                    anno_obj_list[choose_id].center[0]-=delta
                    box_list[choose_id].points = o3d.utility.Vector3dVector(yzx)
                    vis.update_geometry(box_list[choose_id])
                else:
                    for x in range(0, len(box_list)):
                        yzx = np.asarray(box_list[x].points)
                        yzx[:, 2]+=delta
                        box_list[x].points = o3d.utility.Vector3dVector(yzx)
                        vis.update_geometry(box_list[x])
                    x_delta = x_delta + delta
                return False

            def Left(vis):
                global y_delta,delta
                if choose_id>=0:
                    yzx = np.asarray(box_list[choose_id].points)
                    yzx[:, 0]-=delta
                    anno_obj_list[choose_id].center[1]-=delta
                    box_list[choose_id].points = o3d.utility.Vector3dVector(yzx)
                    vis.update_geometry(box_list[choose_id])
                else:
                    for x in range(0, len(box_list)):
                        yzx = np.asarray(box_list[x].points)
                        yzx[:, 0]-=delta
                        box_list[x].points = o3d.utility.Vector3dVector(yzx)
                        vis.update_geometry(box_list[x])
                    y_delta = y_delta - delta
                return False

            def Right(vis):
                global y_delta,delta
                if choose_id>=0:
                    yzx = np.asarray(box_list[choose_id].points)
                    yzx[:, 0]+=delta
                    anno_obj_list[choose_id].center[1]+=delta
                    box_list[choose_id].points = o3d.utility.Vector3dVector(yzx)
                    vis.update_geometry(box_list[choose_id])
                else:
                    for x in range(0, len(box_list)):
                        yzx = np.asarray(box_list[x].points)
                        yzx[:, 0]+=delta
                        box_list[x].points = o3d.utility.Vector3dVector(yzx)
                        vis.update_geometry(box_list[x])
                    y_delta = y_delta + delta
                    return False

            def Up(vis):
                global z_delta,delta
                if choose_id>=0:
                    yzx = np.asarray(box_list[choose_id].points)
                    yzx[:, 1]+=delta
                    anno_obj_list[choose_id].center[2]+=delta
                    box_list[choose_id].points = o3d.utility.Vector3dVector(yzx)
                    vis.update_geometry(box_list[choose_id])
                else:
                    for x in range(0, len(box_list)):
                        yzx = np.asarray(box_list[x].points)
                        yzx[:, 1]+=delta
                        box_list[x].points = o3d.utility.Vector3dVector(yzx)
                        vis.update_geometry(box_list[x])
                    z_delta = z_delta + delta
                return False

            def Down(vis):
                global z_delta,delta
                if choose_id>=0:
                    yzx = np.asarray(box_list[choose_id].points)
                    yzx[:, 1]-=delta
                    anno_obj_list[choose_id].center[2]-=delta
                    box_list[choose_id].points = o3d.utility.Vector3dVector(yzx)
                    vis.update_geometry(box_list[choose_id])
                else:
                    for x in range(0, len(box_list)):
                        yzx = np.asarray(box_list[x].points)
                        yzx[:, 1]-=delta
                        box_list[x].points = o3d.utility.Vector3dVector(yzx)
                        vis.update_geometry(box_list[x])
                    z_delta = z_delta - delta
                return False

            def Choose(vis):
                global choose_id
                choose_id+=1
                if choose_id >= len(box_list):
                    choose_id=0
                print(choose_id)
                color_set = np.asarray(box_list[choose_id].colors)
                color_set[:,1]=0
                color_set[:, 0] = 1
                box_list[choose_id].colors = o3d.utility.Vector3dVector(color_set)
                vis.update_geometry(box_list[choose_id])
                color_set = np.asarray(box_list[choose_id-1].colors)
                color_set[:, 1] = 1
                color_set[:, 0] = 0
                box_list[choose_id-1].colors = o3d.utility.Vector3dVector(color_set)
                vis.update_geometry(box_list[choose_id-1])
                return False

            def OutChoose(vis):
                global choose_id
                color_set = np.asarray(box_list[choose_id].colors)
                color_set[:,1]=1
                color_set[:, 0] = 0
                box_list[choose_id].colors = o3d.utility.Vector3dVector(color_set)
                vis.update_geometry(box_list[choose_id])
                choose_id=-1
                return False

            def DelteObj(vis):
                global choose_id
                if choose_id>=0:
                    vis.remove_geometry(box_list[choose_id])
                    anno_obj_list.pop(choose_id)
                    box_list.pop(choose_id)
                    choose_id-=1
                return False

            def CloseWindow(vis):
                global save,choose_id
                save = -1
                choose_id = -1
                vis.close()
                return False

            def Save(vis):
                global save,choose_id
                save=1
                choose_id = -1
                vis.close()
                return False

            def SaveNothing(vis):
                global save,choose_id
                choose_id = -1
                save = 0
                vis.close()
                return False

            key_to_callback = {}
            key_to_callback[ord("O")] = CloseWindow
            key_to_callback[ord("I")] = AddBox
            key_to_callback[ord("W")] = Forward
            key_to_callback[ord("S")] = Back
            key_to_callback[ord("A")] = Left
            key_to_callback[ord("D")] = Right
            key_to_callback[ord("Q")] = Up
            key_to_callback[ord("E")] = Down
            key_to_callback[ord("C")] = Choose
            key_to_callback[ord("X")] = OutChoose
            key_to_callback[ord("F")] = DelteObj
            key_to_callback[ord("P")] = Save
            key_to_callback[ord("K")] = SaveNothing


            o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback,window_name=cloud_file_path,width=1080, height=1080,left=800, top=25)

        custom_draw_geometry_with_key_callback(pcd)

        if save==1:
            anno_file_path = coor_path + '/' + anno_files[j] + '.txt'
            box_save_list=[]
            for x in range(0,len(anno_obj_list)):
                center = anno_obj_list[x].center
                center[0]-=x_delta
                center[1]+=y_delta
                center[2]+=z_delta
                lwh = anno_obj_list[x].array[3:6]
                roll = np.array([anno_obj_list[x].roll - anno_obj_list[x].coor_roll])
                classes = np.array([anno_obj_list[x].classes])
                box_save=np.concatenate([center,lwh,roll,classes],axis=0).astype(str).tolist()
                box_save=' '.join(box_save)
                box_save_list.append(box_save)
            with open(anno_file_path, 'w') as f:
                for line in box_save_list:
                    f.write(line + '\n')
        elif save==0:
            anno_file_path = coor_path + '/' + anno_files[j] + '.txt'
            with open(anno_file_path, 'w') as f:
                a=0




