#图像处理
import cv2
#实时计算机视觉和机器学习
import mediapipe as mp
#绘图
import matplotlib.pyplot as plt
#复制对象
import copy
#处理时间
import time
#处理3D数据
import open3d
#数值计算
import numpy as np
#显示进度条
from tqdm import tqdm
#管理文件和目录
import os

class my_check:
    #导入solution
    mp_pose = mp.solutions.pose

    #导入绘图函数
    mp_drawing = mp.solutions.drawing_utils

    #导入模型
    img_pose = mp_pose.Pose(static_image_mode=True,      #是否处理静态图像
                            smooth_landmarks=True,       #是否平滑关键点
                            enable_segmentation=True,    #是否人体抠图
                            min_detection_confidence=0.5,#置信度阈值
                            min_tracking_confidence=0.5) #追踪阈值
    
    camera_pose = mp_pose.Pose(static_image_mode=False,      #是否处理静态图像
                                smooth_landmarks=True,       #是否平滑关键点
                                enable_segmentation=True,    #是否人体抠图
                                min_detection_confidence=0.5,#置信度阈值
                                min_tracking_confidence=0.5) #追踪阈值

    video_pose = mp_pose.Pose(static_image_mode=False,   #是否处理静态图像
                            smooth_landmarks=True,       #是否平滑关键点
                            enable_segmentation=True,    #是否人体抠图
                            min_detection_confidence=0.5,#置信度阈值
                            min_tracking_confidence=0.5) #追踪阈值
        


    def __init__(self,input_path=None,img_pose = img_pose,camera_pose=camera_pose,video_pose = video_pose):

        self.flag = 0
        self.img_pose = img_pose
        self.camera_pose = camera_pose
        self.video_pose = video_pose
        if input_path != None:
            self.img = cv2.imread(input_path)
        else:
            self.img = None
        self.results = None

    #图片可视化
    def look_img(self,img=None,figsize=(10,10)):
        if img is None :
            if self.img is None:
                print('未传入图片')
                return
            else:
                img = self.img
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #调整窗口大小(英寸)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    #检测优化
    def optimize(self,img,results,h,w):
        #遍历所有33个关键点,可视化
        for i in range(33):

            #获取该关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)

            radius = 10

            #鼻尖
            if i == 0:
                img = cv2.circle(img,(cx,cy),radius,(0,0,255),-1)
            
            #肩膀
            elif i in [11,12]:
                img = cv2.circle(img,(cx,cy),radius,(223,155,6),-1)
            #髋关节
            elif i in [23,24]:
                img = cv2.circle(img,(cx,cy),radius,(1,240,255),-1)
            #胳臂肘
            elif i in [13,14]:
                img = cv2.circle(img,(cx,cy),radius,(140,47,240),-1)
            #膝盖
            elif i in [25,26]:
                img = cv2.circle(img,(cx,cy),radius,(0,0,255),-1)
            #手腕和脚婉
            elif i in [15,16,27,28]:
                img = cv2.circle(img,(cx,cy),radius,(223,155,60),-1)
            #左手
            elif i in [17,19,21]:
                img = cv2.circle(img,(cx,cy),radius,(94,218,121),-1)
            #右手
            elif i in [18,20,22]:
                img = cv2.circle(img,(cx,cy),radius,(16,144,247),-1)
            #左脚
            elif i in [27,29,31]:
                img = cv2.circle(img,(cx,cy),radius,(29,123,243),-1)
            #右脚
            elif i in [28,30,32]:
                img = cv2.circle(img,(cx,cy),radius,(193,182,255),-1)
            #嘴
            elif i in [9,10]:
                img = cv2.circle(img,(cx,cy),radius,(205,235,255),-1)
            #眼及脸颊
            elif i in [1,2,3,4,5,6,7,8]:
                img = cv2.circle(img,(cx,cy),radius,(94,218,121),-1)
            #其他关键点
            else:
                img = cv2.circle(img,(cx,cy),radius,(0,255,0),-1)
        return img

    #单张图片姿态估计
    def calculate(self):
        self.flag = 1
        if self.img is None:
            print('未传入图片')
            return
        self.outimg = copy.deepcopy(self.img)
        self.outimg_RGB = cv2.cvtColor(self.outimg,cv2.COLOR_BGR2RGB)
        #将RGB图像输入模型,获取预测结果
        if self.results is None:
            self.results = self.img_pose.process(self.outimg_RGB)
        #可视化检测结果
        self.mp_drawing.draw_landmarks(self.outimg,self.results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
        h,w = self.outimg.shape[0],self.outimg.shape[1]
        self.outimg = self.optimize(self.outimg,self.results,h,w)
        self.outimg_RGB = cv2.cvtColor(self.outimg,cv2.COLOR_BGR2RGB)
        self.look_img(self.outimg)

    #三维点云
    def img3D(self):
        self.flag = 1
        if self.img is None:
            print('未传入图片')
            return
        if self.results is None:
            img_RGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
            self.results = self.img_pose.process(img_RGB)
        #创建NumPy数组
        coords = np.array(self.results.pose_landmarks.landmark)

        #在不用耗时循环的条件下,汇总所有点的XYZ坐标
        def get_x(each):
            return each.x
        def get_y(each):
            return each.y
        def get_z(each):
            return each.z

        #分别获取所有关键点的XYZ坐标
        points_x = np.array(list(map(get_x,coords)))
        points_y = np.array(list(map(get_y,coords)))
        points_z = np.array(list(map(get_z,coords)))

        #将三个方向的坐标合并
        points = np.vstack((points_x,points_y,points_z)).T

        #创建Open3D的点云对象
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points)

        #可视化点云
        open3d.visualization.draw_geometries([point_cloud])

    def numpy(self,results=None):
        if results is None:
            if self.results is None:
                if self.img is None:
                    print('未传入图片')
                    return
                else:
                    img_RGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
                    self.results = self.img_pose.process(img_RGB)
            results = self.results
        self.x = results.pose_landmarks
        coords = np.array(results.pose_landmarks.landmark)

        #在不用耗时循环的条件下,汇总所有点的XYZ坐标
        def get_x(each):
            return each.x
        def get_y(each):
            return each.y
        def get_z(each):
            return each.z

        #分别获取所有关键点的XYZ坐标
        points_x = np.array(list(map(get_x,coords)))
        points_y = np.array(list(map(get_y,coords)))
        points_z = np.array(list(map(get_z,coords)))

        #将三个方向的坐标合并
        points = np.vstack((points_x,points_y,points_z)).T
        return points

    #处理单帧的函数
    def process_frame(self,img,pose):

        #记录该帧开始处理的时间
        start_time = time.time()

        h,w = img.shape[0],img.shape[1]

        #BGR转RGB
        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #将RGB图像输入模型,获取预测结果
        results = pose.process(img_RGB)

        #若检测出人体关键点
        if results.pose_landmarks:

            #可视化关键点及骨架连线
            self.mp_drawing.draw_landmarks(img,results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
            
            #优化
            img = self.optimize(img,results,h,w)
        else:
            scaler = 1
            failure_str = 'No Person'
            img = cv2.putText(img,failure_str,(25*scaler,100*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,0),3*scaler)
        
        #记录该帧处理完毕的时间
        end_time = time.time()

        #计算每秒处理图像帧数FPS
        FPS = 1/(end_time - start_time)

        scaler = 1

        #在图像上写FPS数值,参数依次为:图片,添加的文字,左上角坐标,字体,字体大小,颜色,字体粗细
        img = cv2.putText(img,'FPS '+str(int(FPS)),(25*scaler,50*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,0),3*scaler)
        return img
    
    #实时检测
    def detect(self):
        #获取摄像头,传入0表示获取系统默认摄像头
        cap = cv2.VideoCapture(0)

        #无限循环,直到break被触发
        while cap.isOpened():
            #获取画面
            success,frame = cap.read()
            if not success:
                print('Error')
                break

            #!!!处理帧函数
            frame = self.process_frame(frame,self.camera_pose)
            if self.flag:
                #原始图像的宽度和高度分别是 orig_width 和 orig_height  
                orig_width, orig_height = frame.shape[1], frame.shape[0]  
                
                # 想要调整到的最大宽度和高度  
                max_width = 2200  
                max_height = 1200  
                
                # 计算宽度和高度的缩放因子  
                width_ratio = max_width / float(orig_width)  
                height_ratio = max_height / float(orig_height)  
                
                # 选择最小的缩放因子，以确保图像不会失真  
                ratio = min(width_ratio, height_ratio)  
                
                # 计算新的宽度和高度  
                new_width = int(orig_width * ratio)  
                new_height = int(orig_height * ratio)  
                
                # 调整图像大小  
                frame = cv2.resize(frame, (new_width, new_height))
            
            else:
                #原始图像的宽度和高度分别是 orig_width 和 orig_height  
                orig_width, orig_height = frame.shape[1], frame.shape[0]  
                
                # 想要调整到的最大宽度和高度  
                max_width = 1200  
                max_height = 600  
                
                # 计算宽度和高度的缩放因子  
                width_ratio = max_width / float(orig_width)  
                height_ratio = max_height / float(orig_height)  
                
                # 选择最小的缩放因子，以确保图像不会失真  
                ratio = min(width_ratio, height_ratio)  
                
                # 计算新的宽度和高度  
                new_width = int(orig_width * ratio)  
                new_height = int(orig_height * ratio)  
                
                # 调整图像大小  
                frame = cv2.resize(frame, (new_width, new_height))

            #展示处理后的三通道图片
            cv2.imshow('my_window',frame)

            #按键盘上的q或esc退出(在英文输入法下)
            if cv2.waitKey(1) in [ord('q'),27]:
                break

        #关闭摄像头
        cap.release()

        #关闭图像窗口
        cv2.destroyAllWindows()
    
    #视频处理
    def video(self,video_path):
        #使用os.path.basename获取文件名,不带路径
        filehead = os.path.basename(video_path) 
        output_path = "out-" + filehead  
        #获取输出文件所在的目录
        output_dir = os.path.dirname(output_path)

        #确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        print('视频开始处理', video_path)  
    
        #打开视频文件
        cap = cv2.VideoCapture(video_path)  
        #视频帧数
        frame_count = 0  
        #视频大小
        frame_size = None  
        #视频帧率
        fps = None  
        while cap.isOpened():  
            ret, frame = cap.read()  
            if not ret:  
                break  
            frame_count += 1  
            if frame_size is None:  
                frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
            if fps is None:  
                fps = cap.get(cv2.CAP_PROP_FPS)  
        cap.release()  
        print('视频总帧数为', frame_count)  

        #设置视频编码器的四个字符代码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        #创建视频写入对象,参数:输出视频文件路径,编码器类型,视频帧率,视频帧大小
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)  
    
        # 进度条绑定视频总帧数
        with tqdm(total=frame_count) as pbar:  
            #打开视频文件
            cap = cv2.VideoCapture(video_path)  
            try:  
                while cap.isOpened():  
                    ret, frame = cap.read()  
                    if not ret:  
                        break  
    
                    # 处理帧  
                    try:  
                        frame = self.process_frame(frame,self.video_pose)  
                    except Exception as e:  
                        print(f'处理帧时出错: {e}')  
                        # 跳过当前帧，继续处理下一帧  
                        continue
                    
                    #写入每一帧
                    out.write(frame)  

                    #更新进度条
                    pbar.update(1)  
            except Exception as e:  
                print(f'处理视频时出错: {e}')  
            finally:  
                out.release()  
                cap.release()  
    
        print('视频已保存', output_path)
        



