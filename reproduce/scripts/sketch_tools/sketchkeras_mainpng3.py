import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
from keras.models import load_model
import cv2
import glob
from helperpng import *

mod = load_model('mod.h5')

Pout_dir='O1417sketch_out'
if not os.path.exists(Pout_dir): os.mkdir(Pout_dir)

def main():
    for num in ('01','02'):
        in_dir = "O1417/" + num +"/*/"
        out_dir = os.path.join(Pout_dir,num)
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        if not os.path.exists(out_dir + "/enhanced"): os.mkdir(out_dir + "/enhanced")
        if not os.path.exists(out_dir + "/sketchKeras"): os.mkdir(out_dir + "/sketchKeras")
        if not os.path.exists(out_dir + "/raw"): os.mkdir(out_dir + "/raw")
        if not os.path.exists(out_dir + "/pured"): os.mkdir(out_dir + "/pured")
        if not os.path.exists(out_dir + "/colored"): os.mkdir(out_dir + "/colored")
        for types in ('*.png', '*.jpg'):
            for files1 in glob.glob(in_dir + types):
                filepath, filename = os.path.split(files1)
                realname, ex = os.path.splitext(filename)
                # print(realname)
                from_mat = cv2.imread(files1)
                from_mat2 = from_mat
                width = float(from_mat.shape[1])
                height = float(from_mat.shape[0])
                new_width = 0
                new_height = 0
                if (width > height):
                    from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
                    new_width = 512
                    new_height = int(512 / width * height)
                else:
                    from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
                    new_width = int(512 / height * width)
                    new_height = 512
                # cv2.imshow('raw', from_mat)
                from_mat3 = from_mat
                from_mat = from_mat.transpose((2, 0, 1))
                light_map = np.zeros(from_mat.shape, dtype=np.float)
                for channel in range(3):
                    light_map[channel] = get_light_map_single(from_mat[channel])
                light_map = normalize_pic(light_map)
                light_map = resize_img_512_3d(light_map)
                line_mat = mod.predict(light_map, batch_size=1)
                line_mat = line_mat.transpose((3, 1, 2, 0))[0]
                line_mat = line_mat[0:int(new_height), 0:int(new_width), :]

                # show_active_img_and_save('sketchKeras_colored', line_mat, os.path.join(out_dir, 'colored', filename))

                line_mat = np.amax(line_mat, 2)

                if (new_width < 256):
                    flag = 1
                    new_width = 256
                    new_height = int(256 / width * height)
                    from_mat2 = cv2.resize(from_mat2, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(out_dir, 'raw', realname + ".jpg"), from_mat2)
                elif (new_height < 256):
                    flag = 1
                    new_width = int(256 / height * width)
                    new_height = 256
                    from_mat2 = cv2.resize(from_mat2, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(out_dir, 'raw', realname + ".jpg"), from_mat2)
                else:
                    flag = 0
                    cv2.imwrite(os.path.join(out_dir, 'raw', realname + ".jpg"), from_mat3)

                show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat,os.path.join(out_dir, 'enhanced', realname + ".jpg"), flag,new_width, new_height)
                #show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat,os.path.join(out_dir, 'pured', realname + ".jpg"), flag,new_width, new_height)
                show_active_img_and_save_denoise('sketchKeras', line_mat,os.path.join(out_dir, 'sketchKeras', realname + ".jpg"), flag,new_width, new_height)


if __name__ == '__main__':
    main()
