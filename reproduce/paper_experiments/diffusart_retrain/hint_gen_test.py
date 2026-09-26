import numpy as np
import cv2
import skimage

def making_mask_flat_colors_64(mask_simp, color, color_hint, size, hinttype):
    choice = np.random.choice(['width', 'height', 'diag'])

    avg_wht = 0.99

    if hinttype == 'dot':
        mask = np.zeros( (64, 64) )
        
        hh = np.random.randint(64)        
        ww = np.random.randint(64)  

        mask[hh, ww] = 1
        mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)
        mask_simp += mask[:, :, np.newaxis]


        color_lists = color[mask==1]
        avg_color = np.mean(color_lists, axis=0)

        color_hint[mask==1] = avg_color 


        return color_hint, mask_simp         

    else:

        if choice == 'width':
            rnd_height = 1
            rnd_width = np.random.randint(5, 30)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'height':
            rnd_height = np.random.randint(4, 30)
            rnd_width = 1

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'diag':

            thick = 1
            rnd_width = np.random.randint(4, 30)

            rnd1 = np.random.randint(size - thick - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.line(rnd1, rnd2, rnd1+rnd_width, rnd2+rnd_width)
            ii = np.hstack([ii + i for i in range(thick)])
            jj = np.tile(jj, thick)
            scrib_color = color[ii, jj]


        if choice == 'diag':
            count_white = np.sum(np.all(scrib_color == 1, axis = 1))
            count_black = np.sum(np.all(scrib_color == 0, axis = 1))
            size_pix = scrib_color.shape[0]
        else:
            count_white = np.sum(np.all(scrib_color == 1, axis = 2))
            count_black = np.sum(np.all(scrib_color == 0, axis = 2))
            size_pix = scrib_color.shape[0] * scrib_color.shape[1]
        avg_wht = count_white / size_pix
        avg_black = count_black / size_pix



        # if avg_wht < 0.6 and avg_black < 0.6:
        mask_single = np.zeros( (64, 64) )
        mask_simp[ii, jj] = 1
        mask_single[ii,jj] = 1
        mask_simp = cv2.resize(mask_simp, (256, 256), interpolation = cv2.INTER_NEAREST)
        mask_single = cv2.resize(mask_single, (256, 256), interpolation = cv2.INTER_NEAREST)

        color_lists = color[mask_single==1]
        avg_color = np.mean(color_lists, axis=0)

        color_hint[mask_single==1] = avg_color 

        return color_hint, mask_simp


original = cv2.imread("3_color.png")
h, w, _ = original.shape
mask = np.zeros((256, 256, 1))

repeat = np.random.randint(5, 25)
color_hint = np.zeros(original.shape)
for _ in range(repeat):
    color_hint, mask = making_mask_flat_colors_64(mask, original, color_hint, size=64, hinttype='dot')

cv2.imwrite("testtest.png", color_hint)