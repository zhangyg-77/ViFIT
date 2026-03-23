import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

def caijian(path_in, path_out, sub,target, size_w=512, size_h=512, step=256,t = 0):

    ims_list = os.listdir(path_in  ) #在此例中调用时，ims_list为['image.png', 'label.png']
    
    i = t
    #l = 0
    #label_index_dict = {}  # 创建一个空字典用于存储 label 和其对应的索引
    #with open(path_out+'/train.txt', 'r') as text_file:
    for im_list in ims_list:
        # if im_list == '2.jpeg':
            img = cv2.imread(path_in + '/' +im_list)
            print(path_in + '/' + sub +'/' +im_list)
            size = img.shape
            
                # label = im_list.split('-')[2]
                
                # if label not in label_index_dict:  # 如果 label 不在字典中，添加进去
                #     assert label not in label_index_dict
                #     label_index_dict[label] = len(label_index_dict) + 1  # 分配一个唯一索引
                # l = label_index_dict[label]  # 获取 label 对应的索引
                # print(l, label)
                
                # print(label,'_',l,file=text_file)
                

            for h in range(0, size[0], int(step)):
                star_h = h #star_h表示起始高度，从0以步长step=256开始循环
                for w in range(0, size[1], int(step)):
                    star_w = w #star_w表示起始宽度，从0以步长step=256开始循环

                    end_h = star_h + size_h #end_h是终止高度
                    if end_h > size[0]:#如果边缘位置不够512的列
                        # 以倒数512形成裁剪区域
                        star_h = size[0] - size_h
                        end_h = star_h + size_h

                    end_w = star_w + size_w #end_w是中止宽度
                    if end_w > size[1]:#如果边缘位置不够512的行
                        # 以倒数512形成裁剪区域
                        star_w = size[1] - size_w
                        end_w = star_w + size_w
                    
                    cropped = img[int(star_h):int(end_h), int(star_w):int(end_w)]#执行裁剪操作                
                    i = i+1
                    
                    name_img = str(i)
                    """ if not os.path.exists(path_out+'/'+sub):
                        os.makedirs(path_out+'/'+sub) """
                    cv2.imwrite('{}/{}.png'.format(path_out, name_img  ), cropped)#将切割得到的小图片存在path_out路径下    
                    #cv2.imwrite('{}/{}.tif'.format(path_out+'/'+ sub +'B', name_img), cropped)#将切割得到的小图片存在path_out路径下    
    t = i
    print("已将 {} 中的 {} 张大图分割为 {} 张小图\n".format(path_in+'/'+sub, len(ims_list), i))
    return t
if __name__ == '__main__':

    path_in = r'/home/lcy/train/shujuji(qita)/delete2/XF_7.1'               # 输入地址
    path_out = r'/home/lcy/train/pix2pix/datasets/XF_7.1/trainA'     # 输出地址
    sub_path = ['train']
    target = 'A'
    t = 0
    bei = 2
    for sub in sub_path:
        print('正在处理：', path_in+'/'+sub+target)
        i = caijian(path_in, path_out, sub, target, size_w=256*bei, size_h=256*bei, step=256*bei,t=t)
        t = i
    print('结束')
