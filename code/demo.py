from matplotlib import pyplot as plt
import os
import numpy as np
import csv
import my_check
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from PIL import Image,ImageDraw
import sys
import tqdm
import cv2
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

#图片可视化
def show_image(img,figsize=(10,10)):
    '''Shows output PLT image.'''
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

#人体姿态编码
class FullBodyPoseEmbedder():
    '''Converts 3D pose landmarks into 3D embedding.'''

    def __init__(self,torso_size_multiplier=2.5):
        #torso_size_multiplier:身体尺寸乘数
        self._torso_size_multiplier = torso_size_multiplier

        #_landmark_names:关键点名称列表
        self._landmark_names = [
            'nose',
            'left_eye_inner','left_eye','left_eye_outer',
            'right_eye_inner','right_eye','right_eye_outer',
            'left_ear','right_ear',
            'mouth_left','mouth_right',
            'left_shoulder','right_shoulder',
            'left_elbow','right_elbow',
            'left_wrist','right_wrist',
            'left_pinky_1','right_pinky_1',
            'left_index_1','right_index_1',
            'left_thumb_2','right_thumb_2',
            'left_hip','right_hip',
            'left_knee','right_knee',
            'left_ankle','right_ankle',
            'left_heel','right_heel',
            'left_foot_index','right_foot_index'
        ]

    #landmarks:关键点数组
    def __call__(self,landmarks):
        #landmarks.shape[0]:关键点数量
        assert landmarks.shape[0] == len(self._landmark_names),'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

        #Get pose landmarks.
        landmarks = np.copy(landmarks)

        landmarks = self._normalize_pose_landmarks(landmarks)

        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding
        
    #标准化关键点位置
    def _normalize_pose_landmarks(self,landmarks):
        landmarks = np.copy(landmarks)

        #Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        #将关键点平移到该中心
        landmarks -= pose_center

        #Normalize scale.
        pose_size = self._get_pose_size(landmarks,self._torso_size_multiplier)
        #归一化
        landmarks /= pose_size
        #非必须,但方便处理
        landmarks *= 100

        return landmarks
    
    #获取姿态中心点
    def _get_pose_center(self,landmarks):
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center
    
    #获取姿态尺寸
    def _get_pose_size(self,landmarks,torso_size_multiplier):
        
        landmarks = landmarks[:,:2]

        #臀部中点
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        #肩部中点
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        #计算欧几里得距离
        hip_distance = np.linalg.norm(hips - shoulders)  
        pose_size = hip_distance * torso_size_multiplier  
        return pose_size 
    
    def _get_pose_distance_embedding(self, landmarks, keypoint_names=None):  
        if keypoint_names is None:
            keypoint_names = self._landmark_names
        # 初始化一个空列表来存储距离  
        distances = []  
          
        # 遍历关键点名称列表，计算每对关键点之间的距离  
        for i in range(len(keypoint_names) - 1):  
            for j in range(i + 1, len(keypoint_names)):  
                name_from = keypoint_names[i]  
                name_to = keypoint_names[j]  
                distance = self._get_distance_by_names(landmarks, name_from, name_to)  
                distances.append(distance)  
          
        # 将距离列表转换为NumPy数组，形成嵌入向量  
        embedding = np.array(distances)  
        return embedding 
    
    #根据给定的两个关键点名称，计算这两个点的坐标平均值
    def _get_average_by_name(self,landmarks,name_from,name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5
    
    #计算两个关键点之间的欧几里得距离
    def _get_distance_by_names(self,landmarks,name_from,name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from,lmk_to)
    
    #计算两个关键点之间的欧几里得距离
    def _get_distance(self,lmk_from,lmk_to):
        return np.linalg.norm(lmk_to - lmk_from)

#人体姿态分类
class PoseSample():
    
    def __init__(self,name,landmarks,class_name,embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding

class PoseSampleOutlier():

    def __init__(self,sample,detected_class,all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes

class PoseClassifier():

    def __init__(self,
                 pose_samples_folder,      #存放姿态样本的文件路径
                 pose_embedder,            #嵌入器
                 file_extension='csv',     #姿势样本文件的扩展名
                 file_separator=',',       #用于分割CSV文件中数据的字符
                 n_landmarks=33,           #姿势中的关键点数量
                 n_dimensions=3,           #每个关键点的维度数量
                 top_n_by_max_distance=30, #基于最大距离选择的顶部样本数量
                 top_n_by_mean_distance=10,#基于平均距离选择的顶部样本数量
                 axes_weights=(1.,1.,0.2)):#各轴的权重
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension,
                                                     file_separator,
                                                     n_landmarks,
                                                     n_dimensions,
                                                     pose_embedder)
    
    #加载姿态样本
    def _load_pose_samples(self,
                           pose_samples_folder,#存放姿势样本文件的路径
                           file_extension,     #姿势样本文件的扩展名
                           file_separator,     #CSV文件中用于分隔数据的字符
                           n_landmarks,        #关键点的数量
                           n_dimensions,       #每个关键点的维度数量
                           pose_embedder):     #用于嵌入姿势的嵌入器
        
        #遍历所有指定路径下指定扩展名结尾的文件
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            class_name = file_name[:-(len(file_extension)+1)]

            with open(os.path.join(pose_samples_folder,file_name)) as csv_file:
                csv_reader = csv.reader(csv_file,delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions +1,'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:],np.float32).reshape([n_landmarks,n_dimensions])
                    pose_samples.append(PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks)
                    ))

        return pose_samples
    
    #查找姿势样本中的异常值
    def find_pose_sample_outliers(self):
        outliers = []
        for sample in self._pose_samples:
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [class_name for class_name,count in pose_classification.items() if count == max(pose_classification.values())]

            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(sample,class_names,pose_classification))

        return outliers
    
    def __call__(self,pose_landmarks):
        assert pose_landmarks.shape == (self._n_landmarks,self._n_dimensions),'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1,1,1]))

        max_dist_heap = []
        for sample_idx,sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding)*self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding)*self._axes_weights)
            )
            max_dist_heap.append([max_dist,sample_idx])
        
        max_dist_heap = sorted(max_dist_heap,key=lambda x:x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []
        for _,sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding)*self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding)*self._axes_weights)
            )
            mean_dist_heap.append([mean_dist,sample_idx])
        
        mean_dist_heap = sorted(mean_dist_heap,key=lambda x:x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = {class_name:class_names.count(class_name) for class_name in set(class_names)}

        return result

class BootstrapHelper():
  
  def __int__(self,
        images_in_folder,
        images_out_folder,
        csvs_out_folder):
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._csvs_out_folder = csvs_out_folder

    self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder)])

  def bootstrap(self,per_pose_class_limit=None):
    if not os.path.exists(self._csvs_out_folder):
      os.makedirs(self._csvs_out_folder)
    
    for pose_class_name in self._pose_class_names:
      print('Bootstrapping',pose_class_name,file=sys.sstderr)

      images_in_folder = os.path.join(self._images_in_folder,pose_class_name)
      images_out_folder = os.path.join(self._images_out_folder,pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder,pose_class_name + '.csv')
      if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)

      with open(csv_out_path,'w') as csv_out_file:
        csv_out_writer = csv.writer(csv.writer(csv_out_file,delimiter='.',quoting=csv.QUOTE_MINIMAL))
        image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        if per_pose_class_limit is not None:
          image_names = image_names[:per_pose_class_limit]

        for image_name in tqdm(image_names):
          input_frame = cv2.imread(os.path.join(images_in_folder,image_name))
          input_frame = cv2.cvtColor(input_frame,cv2.COLOR_BGR2RGB)

          with mp_pose.Pose(upper_body_only=False) as pose_tracker:
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

          output_frame = input_frame.copy()
          if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
          output_frame = cv2.cvtColor(output_frame,cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(images_out_folder,image_name),output_frame)

          if pose_landmarks is not None:
            frame_height,frame_width = output_frame.shape[0],output_frame.shape[1]
            pose_landmarks = np.array(
                [[lmk.x * frame_width,lmk.y * frame_height,lmk.z * frame_width]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
            assert pose_landmarks.shape == (33,3),'Unexpected landmarks shape:{}'.format(pose_landmarks.shape)
            csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str).tolist())
          
          projection_xz = self._draw_xz_projection(
              output_frame=output_frame,pose_landmarks=pose_landmarks)
          output_frame = np.concatenate((output_frame,projection_xz),axis=1)
        
  def _draw_xz_projection(self,output_frame,pose_landmarks,r=0.5,color='red'):
    frame_height,frame_width = output_frame.shape[0],output_frame.shape[1]
    img = Image.new('RGB',(frame_width,frame_height),color='white')

    if pose_landmarks is None:
      return np.asarray(img)

    r *= frame_width * 0.01

    draw = ImageDraw.Draw(img)
    for idx_1,idx_2 in mp_pose.POSE_CONNEXTIONS:
      x1,y1,z1 = pose_landmarks[idx_1] * [1,1,-1] + [0,0,frame_height * 0.5]
      x2,y2,z2 = pose_landmarks[idx_2] * [1,1,-1] + [0,0,frame_height * 0.5]

      draw.ellipse([x1 - r,z1 - r,x1 + r,z1 + r],fill=color)
      draw.ellipse([x2 - r,z2 - r,x2 + r,z2 + r],fill=color)
      draw.line([x1,z1,x2,z2],width=int(r),fill=color)
    
    return np.asarray(img)

  def align_image_and_csvs(self,print_removed_items=False):

    for pose_class_name in self._pose_class_names:
      images_out_folder = os.path.join(self._images_out_folder,pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder,pose_class_name + '.csv')

      rows = []
      with open(csv_out_path) as csv_out_file:
        csv_out_reader = csv.reader(csv_out_file,delimiter=',')
        for row in csv_out_reader:
          rows.append(row)
        
      image_names_in_csv = []
      
      with open(csv_out_path,'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file,delimiter=',',quoting=csv.QUOTE_MINIMAL)
        for row in rows:
          image_name = row[0]
          image_path = os.path.join(images_out_folder,image_name)
          if os.path.exists(image_path):
            image_names_in_csv.append(image_name)
            csv_out_writer.writerow(row)
          elif print_removed_items:
            print('Removed image from CSV: ',image_path)
          
      for image_name in os.listdir(images_out_folder):
        if image_name not in image_names_in_csv:
          image_path = os.path.join(images_out_folder,image_name)
          os.remove(image_path)
          if print_removed_items:
            print('Removed image from folder: ',image_path)
  
  def analyze_outliers(self,outliers):
    
    for outlier in outliers:
      image_path = os.path.join(self._images_out_folder,outlier.sample.class_name,outlier.sample_name)

      print('Outlier')
      print(' sample path = ',image_path)
      print(' sample class = ',outlier.sample.class_name)
      print(' detected class = ',outlier.detected_class)
      print(' all classes = ',outlier.all_classes)

      img = cv2.imread(image_path)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      show_image(img,figsize=(20,20))
  def remove_outliers(self,outliers):
    for outlier in outliers:
      image_path = os.path.join(self._images_out_folder,outlier.sample.class_name,outlier.sample.name)
      os.remove(image_path)
  
  def print_images_in_statistics(self):
    self._print_images_statistics(self._images_in_folder,self._pose_class_names)
  
  def print_images_out_statistics(self):
    self._print_images_statistics(self._images_out_folder,self._pose_class_names)

  def _print_images_statistics(self,images_folder,pose_class_names):
    print('Number of images per pose class:')
    for pose_class_name in pose_class_names:
      n_images = len([
          n for n in os.listdir(os.path.join(images_folder,pose_class_name))
          if not n.startswith('.')])
      print(' {}:{}'.format[pose_class_name,n_images])
        
