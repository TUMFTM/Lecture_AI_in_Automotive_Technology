
### Import the necessary python libraries

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions
import numpy as np
import os
from moviepy.editor import VideoFileClip
import imageio
imageio.plugins.ffmpeg.download()



def process_image(image):

    imshape = image.shape
    gray = functions.grayscale(image)
    blur_gray = functions.gaussian_blur(gray, 5)
    canny_blur = functions.canny(blur_gray, 100, 200)
    vertices = np.array([[(50, imshape[0]), (450, 320), (500, 320), (900, imshape[0])]], dtype=np.int32)
    region_masked = functions.region_of_interest(canny_blur, vertices)
    hough_picture = functions.hough_lines(region_masked, 2, np.pi / 180, 20, 50, 30)

    result = functions.weighted_img(hough_picture, image)
    return result
    #plt.imshow(gray)
    #plt.show()

################        Process for IMAGES        #######################
image = 0
if image == 1:
    images = os.listdir("test_images/")
    for img_file in images:
        print('Loading an image')
        image = mpimg.imread('test_images/' + img_file)

        print('Processing an image')
        processed_image = process_image(image)

        plt.imshow(processed_image)
        print('Saving an image \n')
        mpimg.imsave('output_images/lines-' + img_file, processed_image)


################        Process for Videos       #######################
video = 1
if video == 1:
    print('Loading a video')
    white_output = 'output_videos/white_output.mp4'
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    a = clip1.fl_image

    print('Processing a video')
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

    print('Saving the video')
    white_clip.write_videofile(white_output, audio=False)



