import os

def get_fileNames(rootdir,type):
    fs = []
    for root,dirs,files in os.walk(rootdir,topdown=True):
        for name in files:
            _,ending = os.path.splitext(name)
            if ending==type:
                fs.append(name.split(".")[0])
    return fs


rootdir = "D:\\360Downloads\\\Smoke-Detect-by-yolov5_v2\\data\\images"
jpglist = set(get_fileNames(rootdir,".jpg"))
txtlist = set(get_fileNames(rootdir,".xml"))

for i in list(jpglist^txtlist):
    i = str(i)+".jpg"
    os.remove(os.path.join(rootdir,i))


