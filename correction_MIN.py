import easygui
from cv2 import EVENT_LBUTTONDBLCLK,circle,getPerspectiveTransform,warpPerspective,INTER_LANCZOS4,imread,imwrite,getWindowProperty,destroyAllWindows,namedWindow,imshow,setMouseCallback,WND_PROP_AUTOSIZE,waitKeyEx
import numpy as np
from copy import deepcopy
import functools
#mouse callback function
def draw_circle(event,x,y,flags,param):
    if event==0:return
    global point_list
    if len(point_list)<4:
        if event==EVENT_LBUTTONDBLCLK:
            point_list.append([x,y])
            circle(img,(x,y),10,(0,0,255),-1)
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
    perspective_matrix = getPerspectiveTransform(src=src_array, dst=dst_array)
    img = warpPerspective(src=bk_img, M=perspective_matrix, dsize=(img.shape[1], img.shape[0]), flags=INTER_LANCZOS4)
    global have_not_trans
    have_not_trans = False

# 创建图像与窗口并将窗口与回调函数绑定
filename = easygui.fileopenbox('选择需要处理的图片','选择图片')
# filename = 'D:\\document\\aff_correction\\20160908221841999.jpg'
img = imread(filename)
have_not_trans = True
bk_img = deepcopy(img)
namedWindow('image')
setMouseCallback('image',draw_circle)
point_list = []
while(1):
    imshow('image',img)
    keycode = waitKeyEx(20)
    if getWindowProperty('image', WND_PROP_AUTOSIZE) < 1:# 使用关闭按钮关闭窗口
        break
    if keycode==-1:continue#显示
    elif keycode&0xFF==27:#esc
        break
    elif keycode&0xFF==13:#回车
        savename = easygui.filesavebox("保存图片","save pic")
        imwrite(savename,img)        
        break
    elif keycode == 2424832:#左
        img=np.rot90(img)
    elif keycode == 2555904:#右
        for _ in range(3):
            img=np.rot90(img)
    elif keycode & 0xFF == ord('r') or keycode & 0xFF == ord('R') or keycode & 0xFF == ord(' '):
        img = deepcopy(bk_img)#图片复原
        point_list = []
        have_not_trans = True
destroyAllWindows()