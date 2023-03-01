from google.colab import drive
drive.mount('/content/drive')
import os
import glob
import cv2
import time
import glob
import random

dolphin_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/dolphin/*')
shark_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/shark/*')
whale_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/whale/*')

# dic = {'dolphin':dolphin_img_list, 'shark': shark_img_list, 'whale': whale_img_list}
# for key in dic.keys():
#   print(f'{key}이미지가 ',len(dic[key]), '개 있습니다.')
# print('------------------------------------------------------------------------')

# length_list = []
# for key in dic.keys():
#   print(f'{key}이미지는 trian, valid, test셋에 대해 ',int(len(dic[key])*0.7), int(len(dic[key])*0.2), int(len(dic[key])*0.1), '개씩 배정해주세요.')
#   length_list.append([int(len(dic[key])*0.7), int(len(dic[key])*0.2), int(len(dic[key])*0.1)])
# dolphin_img_list

class Make_dataset_dir():
  def __init__(self, root_dir):
    self.root_path = root_dir+'/' if root_dir[-1] != '/' else root_dir # 현재 진행할 프로젝트 -> root path는 /content/drive/MyDrive/CV_seminar_project/ 가 되어야합니다.
    self.img_path_list = root_dir+'original' # 전달한 이미지들의 상위 경로
    self.trainset_path = root_dir+'train/'
    self.validset_path = root_dir+'valid/'
    self.testset_path = root_dir+'test/'
    self.class_list = ['dolphin', 'shark', 'whale']

  def mk_dir(self):
    os.mkdir(self.trainset_path)
    os.mkdir(self.validset_path)
    os.mkdir(self.testset_path)
    for name in self.class_list:
      os.mkdir(self.trainset_path + name)
      os.mkdir(self.validset_path + name)
      os.mkdir(self.testset_path + name)

  def move_img(self):
    for animal in [dolphin_img_list, shark_img_list, whale_img_list]:
      random.shuffle(animal)
      n_train = int(len(animal)*0.7) 
      n_valid = int(len(animal)*0.2)
      n_test = int(len(animal)*0.1)
      # n_test +=len(animal) - (n_train + n_valid + n_test) 
      print("{}, {}, {} 합:{}, 원래길이:{}".format(n_train, n_valid, n_test, n_train + n_valid + n_test, len(animal)))
      num = [0,n_train ,n_train+n_valid, n_train + n_valid + n_test]
      for i, var in enumerate(["train", "valid", "test"]):
        for file in animal[num[i]:num[i+1]]:
          to_file = file.split('/')
          to_file[5] = var
          to_file="/".join(to_file)
          os.rename(file, to_file)

  def run(self):
    start = time.time()
    self.mk_dir()
    self.move_img()
    print('총 소요시간: ', time.time()-start)

  def checking_dirs(self):
    animals = ["dolphin", "shark", "whale"]
    datatypes = ["train", "valid", "test"]
    for anis in animals:
      for datatype in datatypes:
        list_fp = ['','content','drive','MyDrive','CV_seminar_project','데이터 종류','동물','*']
        list_fp[5] = datatype
        list_fp[6] = anis
        fp = "/".join(list_fp)
        ani_img_list = glob.glob(fp)
        print("동물: {}, 데이터 종류: {}, 데이터 크기: {}".format(anis, datatype,len(ani_img_list)))
