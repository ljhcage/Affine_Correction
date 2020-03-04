import easygui
import cv2
import numpy as np
from copy import deepcopy
import functools
import sys
import os
#mouse callback function
def draw_circle(event,x,y,flags,param):
    if event==0:return
    global point_list
    if len(point_list)<4:
        if event==cv2.EVENT_LBUTTONDBLCLK:
            point_list.append([x,y])
            cv2.circle(img,(x,y),radious,(0,0,255),-1)
    if len(point_list)==4 and have_not_trans:
        aff_correction()
        
def add_sort(point1,point2):
    if point1[0]+point1[1]<point2[0]+point2[1]:
        return -1
    elif point1[0]+point1[1]==point2[0]+point2[1]:
        if img.shape[0]>img.shape[1]:
            if point1[0]<point2[0]:return -1
            else:return 1
        else:
            if point1[1]<point1[1]:return -1
            else:return 1
    else:return 1

def aff_correction():
    '''max_x = max([point[0] for point in point_list])
    min_x = min([point[0] for point in point_list])
    max_y = max([point[1] for point in point_list])
    min_y = min([point[1] for point in point_list])
    '''
    sort_list = sorted(point_list,key=functools.cmp_to_key(add_sort))
    #左上
    A_point = sort_list.pop(0)
    #右下
    D_point = sort_list.pop(-1)
    #B右上 C左下
    if sort_list[0][0]>sort_list[1][0]:
        B_point = sort_list[1]
        C_point = sort_list[0]
    else:B_point,C_point = sort_list[0],sort_list[1]

    global img
    src_array = np.array([A_point,B_point,C_point,D_point],dtype=np.float32)
    dst_array = np.array([[0,0],[0,img.shape[0]],[img.shape[1],0],[img.shape[1],img.shape[0]]],dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(src=src_array, dst=dst_array)
    img = cv2.warpPerspective(src=bk_img, M=perspective_matrix, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4)
    global have_not_trans
    have_not_trans = False

# 创建图像与窗口并将窗口与回调函数绑定
filename = easygui.fileopenbox('选择需要处理的图片','选择图片')
# filename = 'D:\\document\\aff_correction\\20160908221841999.jpg'
while filename:
    os.chdir(os.path.split(filename)[0])
    img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
    have_not_trans = True
    bk_img = deepcopy(img)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL )
    cv2.setMouseCallback('image',draw_circle)
    ori_size = img.shape[0:2]
    max_width = 1280
    max_height = 720
    radious = max(ori_size)//80
    width = min([ori_size[1],max_width])
    height = ori_size[0]/ori_size[1]*width
    if height>max_height:
        height = max_height
        width = ori_size[1]/ori_size[0]*height
    point_list = []
    while(1):
        if have_not_trans:cv2.resizeWindow('image', int(width), int(height))
        cv2.imshow('image',img)
        keycode = cv2.waitKeyEx(20)
        if cv2.getWindowProperty('image', cv2.WND_PROP_AUTOSIZE) < 0:# 使用关闭按钮关闭窗口
            break
        if keycode==-1:continue#显示
        elif keycode&0xFF==27:#esc
            break
        elif keycode&0xFF==13:#回车
            savename = easygui.filesavebox("保存图片","save pic",filename,f"*.{filename.split('.')[-1]}")
            ratio = cv2.getWindowProperty('image', cv2.WND_PROP_ASPECT_RATIO)
            imgsave = cv2.resize(img,(max([img.shape[1],int(img.shape[0]*ratio)]),max([img.shape[0],int(img.shape[1]//ratio)])),cv2.INTER_LANCZOS4)
            cv2.imwrite(savename,imgsave)        
            break
        elif keycode == 2424832:#左
            width,height = height,width
            img=np.rot90(img)
        elif keycode == 2555904:#右
            width,height = height,width
            for _ in range(3):
                img=np.rot90(img)
        elif keycode & 0xFF == ord('r') or keycode & 0xFF == ord('R') or keycode & 0xFF == ord(' '):
            img = deepcopy(bk_img)#图片复原
            point_list = []
            have_not_trans = True
    cv2.destroyAllWindows()
    filename = easygui.fileopenbox('选择需要处理的图片','选择图片')