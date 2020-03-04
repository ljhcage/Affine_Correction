import easygui
import cv2
import numpy as np
from copy import deepcopy
import functools
import sys
import time
import os
#mouse callback function
def draw_circle(event,x,y,flags,param):
    global point_list,is_down,x_base,y_base,temp_point
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        if is_down == 0: 
            is_down = 1
            for index in range(len(point_list)):
                if (point_list[index][0]-x)**2+(point_list[index][1]-y)**2<=radious**2:
                    point_list.pop(index)
                    temp_point = [x,y]
            # x_base,y_base = x, y  # 使鼠标移动距离都是相对于初始点击位置，而不是相对于上一位置
    if event==cv2.EVENT_RBUTTONDBLCLK:
        if len(point_list)<4:
            point_list.append([x,y])
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) and is_down:
        if temp_point:temp_point = [x,y]
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        if temp_point:
            point_list.append(temp_point)
            temp_point = None
        is_down = 0
    """if have_not_trans:cv2.resizeWindow('image', int(width), int(height))
    actual_draw()
    cv2.imshow('image',img)"""
        
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

def cross(point1,point2):
    return (point1[0]*point2[1]+point1[1]*point2[0])

def kua(A,B,C,D):

    if ((((A[0] - B[0])*(D[1] - B[1]) - (A[1] - B[1])*(D[0] - B[0]))*
        ((C[0] - B[0])*(D[1] - B[1]) - (C[1] - B[1])*(D[0] - B[0]))) > 0 or 
        (((B[0] - A[0])*(C[1] - A[1]) - (B[1] - A[1])*(C[0] - A[0]))*
        ((D[0] - A[0])*(C[1] - A[1]) - (D[1] - A[1])*(C[0] - A[0]))) > 0):
        return False
    return True
    '''
    AB,DB,CA,DA = [0]*2,[0]*2,[0]*2,[0]*2    
    AB[0]=A[0]-B[0]
    AB[1]=A[1]-B[1]
    DB[0]=D[0]-B[0]
    DB[1]=D[1]-B[1]
    CA[0]=C[0]-A[0]
    CA[1]=C[1]-A[1]
    DA[0]=D[0]-A[0]
    DA[1]=D[1]-A[1]
    if(cross(AB,DB)*cross(DB,CA)>=0):
        AB[1]=-AB[1];AB[0]=-AB[0]; 
        if(cross(AB,CA)*cross(CA,DA)>=0):
            return 1
        else:
            return 0
    else:
        return 0'''

def is_cross(A_point,B_point,C_point,D_point):
    """
    判断AC和BD是否相交
    """
    if not ( min(A_point[0],C_point[0]) >= max(B_point[0],D_point[0]) or min(B_point[0],D_point[0]) >= max(A_point[0],C_point[0]) or
            min(A_point[1],C_point[1]) >= max(B_point[1],D_point[1]) or min(B_point[1],D_point[1]) >= max(A_point[1],C_point[1])):
            if kua(A_point,B_point,C_point,D_point):return True
    else:return False

def actual_draw():
    if not have_not_trans:return
    if len(point_list)==0 and not temp_point:return
    global img
    img = deepcopy(bk_img)

    if len(point_list)==4 or (len(point_list)==3 and temp_point):
        point_list_to_draw = deepcopy(point_list)
        if temp_point:point_list_to_draw.append(temp_point)
        sort_list = sorted(point_list_to_draw,key=functools.cmp_to_key(add_sort))
        #左上
        A_point = tuple(sort_list.pop(0))
        #右下
        D_point = tuple(sort_list.pop(-1))
        #B右上 C左下
        if sort_list[0][0]>sort_list[1][0]:
            B_point = tuple(sort_list[1])
            C_point = tuple(sort_list[0])
        else:B_point,C_point = tuple(sort_list[0]),tuple(sort_list[1])

        # AC BD 是否相交，相交的话交换CD
        if is_cross(A_point,B_point,C_point,D_point):
            C_point,D_point = D_point,C_point
        if is_cross(A_point,C_point,B_point,D_point):
            B_point,D_point = D_point,B_point
        

        cv2.line(img,A_point,C_point,(238,32,218),max(2,radious//2))
        cv2.line(img,D_point,B_point,(238,32,218),max(2,radious//2))
        cv2.line(img,D_point,C_point,(238,32,218),max(2,radious//2))
        cv2.line(img,A_point,B_point,(238,32,218),max(2,radious//2))
        """cv2.putText(img,f'A{A_point}',A_point, font, 1,(255,255,255),2)#大小，宽度
        cv2.putText(img,f'C{C_point}',C_point, font, 1,(255,255,255),2)
        cv2.putText(img,f'B{B_point}',B_point, font, 1,(255,255,255),2)
        cv2.putText(img,f'D{D_point}',D_point, font, 1,(255,255,255),2)"""
    for x,y in point_list:
        cv2.circle(img,(x,y),radious,(0,0,255),-1)
    if temp_point:
        cv2.circle(img,(temp_point[0],temp_point[1]),radious,(0,215,0),-1)

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

        # AC BD 是否相交，相交的话交换CD
    if is_cross(A_point,B_point,C_point,D_point):
        C_point,D_point = D_point,C_point
    if is_cross(A_point,C_point,B_point,D_point):
        B_point,D_point = D_point,B_point

    global img
    src_array = np.array([A_point,B_point,C_point,D_point],dtype=np.float32)
    dst_array = np.array([[0,0],[0,img.shape[0]],[img.shape[1],0],[img.shape[1],img.shape[0]]],dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(src=src_array, dst=dst_array)
    img = cv2.warpPerspective(src=bk_img, M=perspective_matrix, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4)
    global have_not_trans
    have_not_trans = False

def get4Contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # edged = cv2.Canny(dilate, 75, 200)
    edged = cv2.Canny(dilate, 30, 120, 3)
 
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] 
    docCnt = None
 
    # 确保至少找到一个轮廓
    if len(cnts) > 0:
        # 按轮廓大小降序排列
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # 近似轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果我们的近似轮廓有四个点，则确定找到了纸
            if len(approx) == 4:
                docCnt = approx
                return docCnt

# 创建图像与窗口并将窗口与回调函数绑定
print('''
欢迎使用图片矫正工具
右键双击选点，选择好四个点之后会自动出现四边形轮廓，左键选中选择的角点可拖拽进行调整
确认选点之后按回车或者S键开始进行矫正计算
对矫正计算不满意或在选点过程中都可通过按空格还原初始状态
左方向键或者A键可逆时针旋转图片，右方向键或者D键可顺时针旋转图片，上方向键和W键可上下翻转图片，下方向键和Q可水平翻转图片
进行矫正计算之后可自由拖动窗口大小以改变图片的宽高比
再次按回车或者S键可保存图片
完成一次工作之后软件会自动请求下一张图片，如结束工作直接关闭即可
''')
filename = easygui.fileopenbox('选择需要处理的图片','选择图片')
# filename = 'D:\\document\\Affine_Correction\\20160908221841999.jpg'
font=cv2.FONT_HERSHEY_SIMPLEX
while filename:
    os.chdir(os.path.split(filename)[0])
    img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
    have_not_trans = True
    temp_point = None
    bk_img = deepcopy(img)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL )
    cv2.setMouseCallback('image',draw_circle)
    ori_size = img.shape[0:2]
    max_width = 1280
    max_height = 720
    is_down = 0#一个阶段的鼠标按键
    x_base,y_base = 0,0
    radious = max(ori_size)//80
    width = min([ori_size[1],max_width])
    height = ori_size[0]/ori_size[1]*width
    if height>max_height:
        height = max_height
        width = ori_size[1]/ori_size[0]*height
    # test = get4Contours(deepcopy(bk_img)) #边缘提取暂时还不可用
    # point_list = [list(item[0]) for item in test]
    point_list = []
    while(1):
        if have_not_trans:cv2.resizeWindow('image', int(width), int(height))
        actual_draw()
        cv2.imshow('image',img)
        keycode = cv2.waitKeyEx(50)
        if cv2.getWindowProperty('image', cv2.WND_PROP_AUTOSIZE) < 0:# 使用关闭按钮关闭窗口
            break
        if keycode==-1:continue#显示
        elif keycode&0xFF==27:#esc
            break
        elif keycode&0xFF in [13,ord('S'),ord('s')]:#回车和S
            if len(point_list)==4 and have_not_trans:
                aff_correction()
            elif not have_not_trans:
                savename = easygui.filesavebox("保存图片","save pic",filename,f"*.{filename.split('.')[-1]}")
                if not savename:break
                print(f'{filename}\t->\t{savename}')
                ratio = cv2.getWindowProperty('image', cv2.WND_PROP_ASPECT_RATIO)
                imgsave = cv2.resize(img,(max([img.shape[1],int(img.shape[0]*ratio)]),max([img.shape[0],int(img.shape[1]//ratio)])),cv2.INTER_LANCZOS4)
                cv2.imwrite(savename,imgsave)        
                break
        elif keycode == 2424832 or keycode&0xff==ord('a') or keycode&0xff==ord('A'):#左
            width,height = height,width
            img=np.rot90(img)
        elif keycode == 2555904 or keycode&0xff==ord('d') or keycode&0xff==ord('D'):#右r
            width,height = height,width
            for _ in range(3):
                img=np.rot90(img)
        elif keycode == 2490368 or keycode&0xff in [ord('w'),ord('W')]:#上 W 上下翻转
            img = cv2.flip(img, 0)
        elif keycode == 2621440 or keycode&0xff in [ord('w'),ord('W')]:#下 Q 左右翻转
            img = cv2.flip(img, 1)
        elif keycode & 0xFF in [ord('r'),ord('R'), ord(' ')]:
            img = deepcopy(bk_img)#图片复原
            point_list = []
            have_not_trans = True
            print('restore')
    cv2.destroyAllWindows()
    filename = easygui.fileopenbox('选择需要处理的图片','选择图片')