#importing Required Libraries to run code

import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
import streamlit as st
import geocoder
import os
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler

# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.button import Button
# from kivy.uix.textinput import TextInput
import time

st.set_page_config(page_title="COVID-19 Web App", initial_sidebar_state="expanded", )


def main():
    st.sidebar.title("The output images of your ride will be displayed here.")
    st.title("Smart Ventilation for Vehicles")
    user_input = st.number_input("Before starting, enter the number of occupants inside the vehicle", max_value=8, min_value=1, value=1)
    if st.button('Start!'):
        def ExtractColorHistogram(image, nbins=32, bins_range=(0, 255), resize=None):
            if (resize != None):
                image = cv2.resize(image, resize)
            zero_channel = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
            first_channel = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
            second_channel = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
            return zero_channel, first_channel, second_channel

        # Find Center of the bin edges
        def FindBinCenter(histogram_channel):
            bin_edges = histogram_channel[1]
            bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
            return bin_centers

        # Extracting Color Features from bin lengths
        def ExtractColorFeatures(zero_channel, first_channel, second_channel):
            return np.concatenate((zero_channel[0], first_channel[0], second_channel[0]))

        def GetFeaturesFromHog(image, orient, cellsPerBlock, pixelsPerCell, visualize=False, feature_vector_flag=True):
            if (visualize == True):
                hog_features, hog_image = hog(image, orientations=orient,
                                              pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                                              cells_per_block=(cellsPerBlock, cellsPerBlock),
                                              visualize=True, feature_vector=feature_vector_flag)
                return hog_features, hog_image
            else:
                hog_features = hog(image, orientations=orient,
                                   pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                                   cells_per_block=(cellsPerBlock, cellsPerBlock),
                                   visualize=False, feature_vector=feature_vector_flag)
                return hog_features


        def ExtractFeatures(images, orientation, cellsPerBlock, pixelsPerCell, convertColorspace=False):
            featureList = []
            imageList = []
            for image in images:
                if (convertColorspace == True):
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                local_features_1 = GetFeaturesFromHog(image[:, :, 0], orientation, cellsPerBlock, pixelsPerCell, False,
                                                      True)
                local_features_2 = GetFeaturesFromHog(image[:, :, 1], orientation, cellsPerBlock, pixelsPerCell, False,
                                                      True)
                local_features_3 = GetFeaturesFromHog(image[:, :, 2], orientation, cellsPerBlock, pixelsPerCell, False,
                                                      True)
                x = np.hstack((local_features_1, local_features_2, local_features_3))
                featureList.append(x)
            return featureList




        scaler = StandardScaler()

        import joblib

        classifierSG = joblib.load('root_dir' + 'model.pkl')

        # Sliding Window
        # function to draw sliding Windows

        import matplotlib.image as mpimg

        def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
            # Make a copy of the image
            imcopy = np.copy(img)
            # Iterate through the bounding boxes

            for bbox in bboxes:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                color = (r, g, b)
                # Draw a rectangle given bbox coordinates
                cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
            # Return the image copy with boxes drawn
            return imcopy

        # function to find the windows on which we are going to run the classifier

        def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                         xy_window=(64, 64), xy_overlap=(0.9, 0.9)):
            if x_start_stop[0] == None:
                x_start_stop[0] = 0
            if x_start_stop[1] == None:
                x_start_stop[1] = img.shape[1]
            if y_start_stop[0] == None:
                y_start_stop[0] = 0
            if y_start_stop[1] == None:
                y_start_stop[1] = img.shape[0]

            window_list = []
            image_width_x = x_start_stop[1] - x_start_stop[0]
            image_width_y = y_start_stop[1] - y_start_stop[0]

            windows_x = np.int(1 + (image_width_x - xy_window[0]) / (xy_window[0] * xy_overlap[0]))
            windows_y = np.int(1 + (image_width_y - xy_window[1]) / (xy_window[1] * xy_overlap[1]))

            modified_window_size = xy_window
            for i in range(0, windows_y):
                y_start = y_start_stop[0] + np.int(i * modified_window_size[1] * xy_overlap[1])
                for j in range(0, windows_x):
                    x_start = x_start_stop[0] + np.int(j * modified_window_size[0] * xy_overlap[0])

                    x1 = np.int(x_start + modified_window_size[0])
                    y1 = np.int(y_start + modified_window_size[1])
                    window_list.append(((x_start, y_start), (x1, y1)))
            return window_list

        # function that returns the refined Windows
        # From Refined Windows we mean that the windows where the classifier predicts the output to be a car

        def DrawCars(image, windows, converColorspace=False):
            refinedWindows = []
            for window in windows:

                start = window[0]
                end = window[1]
                clippedImage = image[start[1]:end[1], start[0]:end[0]]

                if (clippedImage.shape[1] == clippedImage.shape[0] and clippedImage.shape[1] != 0):

                    clippedImage = cv2.resize(clippedImage, (64, 64))

                    f1 = ExtractFeatures([clippedImage], 9, 2, 16, converColorspace)

                    # predictedOutput = classifier1.predict([f1[0]])
                    predictedOutput = classifierSG.predict([f1[0]])
                    if (predictedOutput == 1):
                        refinedWindows.append(window)

            return refinedWindows

        # trying out SubSampling using HOG but not able to go through as feature size is not the same.

        def DrawCarsOptimised(image, image1, image2, windows, converColorspace=False):
            refinedWindows = []
            for window in windows:

                start = window[0]
                end = window[1]
                clippedImage = image[start[1]:end[1], start[0]:end[0]]
                clippedImage1 = image1[start[1]:end[1], start[0]:end[0]]
                clippedImage2 = image2[start[1]:end[1], start[0]:end[0]]

                if (clippedImage.shape[1] == clippedImage.shape[0] and clippedImage.shape[1] != 0):

                    clippedImage = cv2.resize(clippedImage, (64, 64)).ravel()
                    clippedImage1 = cv2.resize(clippedImage1, (64, 64)).ravel()
                    clippedImage2 = cv2.resize(clippedImage2, (64, 64)).ravel()

                    # f1=ExtractFeatures([clippedImage], 9 , 2 , 16,converColorspace)
                    f1 = np.hstack((clippedImage, clippedImage1, clippedImage2))
                    f1 = scaler.transform(f1.reshape(1, -1))
                    print(f1.shape)
                    # predictedOutput = classifier1.predict([f1[0]])
                    predictedOutput = classifierSG.predict([f1[0]])
                    if (predictedOutput == 1):
                        refinedWindows.append(window)

            return refinedWindows

        # Determine speed of vehicle

        # Define code for Ventilation Mode

        photonum = 1
        recirc = 1
        CO2 = 500
        totalT = 0 # time since Recirculation Mode = ON
        totalTV = 0 # time since ventilation ON (Recirculation Mode = OFF)

        num = 1

        numfiles = 0
        for files in os.listdir('root_dir' + 'images/'):
            numfiles += 1
        print(numfiles)
        while num <= numfiles:
            if photonum == 1:
                geo = geocoder.ip('me')
                lat1 = geo.lat
                lng1 = geo.lng
                T = 0
                TV = 0
                start_time = time.time()
                # video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                # check, frame = video.read()
                #
                # cv2.imwrite('root_dirscreenshot1.jpg', frame)
                # cv2.waitKey(0)
                # video.release()
                # cv2.destroyAllWindows()

                decvar = 1  # the decvar is used to help define the myint(Number) values to number of vehicles
                image = mpimg.imread('root_dir' + 'images/' + str(num) + '.jpg')

            elif photonum == 2:
                # video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                # check, frame = video.read()
                #
                # cv2.imwrite('root_dirscreenshot2.jpg', frame)
                # cv2.waitKey(0)
                # video.release()
                # cv2.destroyAllWindows()

                decvar = 2
                image = mpimg.imread('root_dir' + 'images/' + str(num) + '.jpg')

            elif photonum == 3:
                # video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                # check, frame = video.read()
                #
                # cv2.imwrite('root_dirscreenshot3.jpg', frame)
                # cv2.waitKey(0)
                # video.release()
                # cv2.destroyAllWindows()

                decvar = 3
                image = mpimg.imread('root_dir' + 'images/' + str(num) + '.jpg')

            elif photonum == 4:
                # video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                # check, frame = video.read()
                #
                # cv2.imwrite('root_dirscreenshot4.jpg', frame)
                # cv2.waitKey(0)
                # video.release()
                # cv2.destroyAllWindows()

                decvar = 4
                image = mpimg.imread('root_dir' + 'images/' + str(num) + '.jpg')

            elif photonum == 5:
                # video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                # check, frame = video.read()
                #
                # cv2.imwrite('root_dirscreenshot5.jpg', frame)
                # cv2.waitKey(0)
                # video.release()
                # cv2.destroyAllWindows()

                decvar = 5
                image = mpimg.imread('root_dir' + 'images/' + str(num) + '.jpg')

            # Sliding Windows
            num += 1
            windows1 = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 464],
                                    xy_window=(64, 64), xy_overlap=(0.15, 0.15))
            windows4 = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 480],
                                    xy_window=(80, 80), xy_overlap=(0.2, 0.2))
            windows2 = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 612],
                                    xy_window=(96, 96), xy_overlap=(0.3, 0.3))
            windows3 = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 660],
                                    xy_window=(128, 128), xy_overlap=(0.2, 0.2))

            windows = windows1 + windows2 + windows3 + windows4

            refinedWindows = DrawCars(image, windows, True)

            f, axes = plt.subplots(2, 1, figsize=(30, 15))

            window_img = draw_boxes(image, windows)

            axes[0].imshow(window_img)
            axes[0].set_title("Window Coverage")

            window_img = draw_boxes(image, refinedWindows)

            axes[1].set_title("Test Image with Refined Sliding Windows")
            axes[1].imshow(window_img)

            # Applying Heatmap

            # function to increase the pixel by one inside each box

            def add_heat(heatmap, bbox_list):
                # Iterate through list of bboxes
                for box in bbox_list:
                    # Add += 1 for all pixels inside each bbox
                    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

                # Return updated heatmap
                return heatmap

            # applying a threshold value to the image to filter out low pixel cells

            def apply_threshold(heatmap, threshold):
                # Zero out pixels below the threshold
                heatmap[heatmap <= threshold] = 0
                # Return thresholded map
                return heatmap

            # find pixels with each car number and draw the final bounding boxes

            from scipy.ndimage.measurements import label

            def draw_labeled_bboxes(img, labels):
                # Iterate through all detected cars
                for car_number in range(1, labels[1] + 1):
                    # Find pixels with each car_number label value
                    nonzero = (labels[0] == car_number).nonzero()
                    # Identify x and y values of those pixels
                    nonzeroy = np.array(nonzero[0])
                    nonzerox = np.array(nonzero[1])
                    # Define a bounding box based on min/max x and y
                    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
                    # Draw the box on the image
                    cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
                # Return the image
                return img

            # testing our heat function

            heat = np.zeros_like(image[:, :, 0]).astype(np.float)

            heat = add_heat(heat, refinedWindows)

            # Apply threshold to help remove false positives
            heat = apply_threshold(heat, 1)

            # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            new_p = Image.fromarray(heatmap)
            if new_p.mode != 'RGB':
                new_p = new_p.convert('RGB')

            labels = label(heatmap)
            hello = draw_labeled_bboxes(image, labels)
            st.sidebar.image(hello, width=270)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            print(" Number of Cars found - ", labels[1])

            if photonum == 5:
                photonum = 0

            photonum += 1
            # below statements are necessary to clear buffer and not get problem in image buffer
            plt.clf()
            plt.close()

            myint = labels[1]

            if decvar == 1:
                myint1 = myint
                devar = 0
            elif decvar == 2:
                myint2 = myint
            elif decvar == 3:
                myint3 = myint
            elif decvar == 4:
                myint4 = myint
            elif decvar == 5:
                myint5 = myint
                devar = 1

            if devar == 1:
                geo2 = geocoder.ip('me')
                lat2 = geo2.lat
                lng2 = geo2.lng
                # print(lat2)
                # print(lng2)

                import math

                a = math.sin(math.radians(lat1)) * math.sin(math.radians(lat2))
                b = math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
                c = math.cos(math.radians(lng2 - lng1))
                distance = math.acos(a + (b * c)) * (6371000 * 3.28084)  # the distance between the two gps coordinates

                speed = distance / (time.time() - start_time)
                traffic = (myint1 + myint2 + myint3 + myint4 + myint5) / 5  # the average number of cars detected in the last five rotation

                if traffic < 2:
                    traffic = 0  # low or no traffic
                elif 2 <= traffic <= 4:
                    traffic = 1  # medium traffic
                else:
                    traffic = 2  # high traffic

                if speed < 10:
                    speedflag = 0  # low speed
                elif 10 <= speed <= 45:
                    speedflag = 1  # medium speed
                else:
                    speedflag = 2  # high speed

                r = 2
                y = 1
                g = 0

                # Outside Air Acceptability function
                if speedflag == 0:
                    osa = r
                elif speedflag == 1 and traffic == 0:
                    osa = g
                elif speedflag == 1 and traffic == 1:
                    osa = y
                elif speedflag == 1 and traffic == 2:
                    osa = r
                elif speedflag == 2 and traffic == 0:
                    osa = g
                elif speedflag == 2 and traffic == 1:
                    osa = g
                elif speedflag == 2 and traffic == 2:
                    osa = y

                N = user_input # number of passengers inside the vehicle
                H = 80  # human exhalation rate in ppm
                V = 800  # ventilation rate of vehicle in ppm

                if recirc == 0:
                    TV = time.time() - start_time
                    T = 0
                    totalT = 0

                elif recirc == 1:
                    T = time.time() - start_time
                    TV = 0
                    totalTV = 0

                totalT += T
                totalTV += TV

                if T == 0:  # calculates Co2 when outside air
                    CO2 += (H * TV / 60 * N) - (V * (TV / 60))
                else:  # calculates Co2 when recirculation
                    CO2 += (H * N * (T / 60)) - (V * (TV / 60))

                # Decision function
                if recirc == 0 and CO2 >= 1000:
                    recirc = 0
                else:
                    if CO2 < 1000:
                        recirc = 1
                    elif osa == g and 1000 <= CO2 <= 2000:
                        if myint3 - myint1 > 0 and myint5 - myint3 > 0:
                            recirc = 0
                        else:
                            recirc = 1
                    elif osa == y and 1500 <= CO2 <= 2000:
                        if myint3 - myint1 >= 0 and myint5 - myint3 >= 0:
                            recirc = 0
                        else:
                            recirc = 1
                    elif osa == r and 1000 <= CO2 <= 2000:
                        recirc = 1
                    elif osa == g and CO2 > 2000:
                        recirc = 0
                    elif osa == y and CO2 > 2000:
                        recirc = 0
                    elif osa == r and CO2 > 2000:
                        if CO2 < 3000:
                            recirc = 1
                        else:
                            recirc = 0

                if recirc == 1:
                    vent = "Turn recirculation mode on."
                elif recirc == 0:
                    vent = "Turn outside air mode on."



                if osa == r:
                    word = 'Bad'
                elif osa == y:
                    word = "Okay"
                else:
                    word = 'Good'
                bigline = ("Traffic " + str(traffic) + " OSA " + str(osa) + " Recirc " + str(recirc) + " totalT " + str(round(totalT)) + " totalTV " + str(
                round(totalTV)) + " CO2 " + str(round(CO2)))
                print(bigline)
                st.write("Traffic Condition: " + str(traffic))
                st.write("Vehicle Speed (mph): " + str(speed))
                st.write("Outside Air Acceptability: " + str(word))
                st.write("CO2 Level (ppm): " + str(round(CO2)))
                st.write('')
                st.write("Total Time since Outside Air Mode On (sec): " + str(round(totalTV)))
                st.write("Total Time since Recirculation Mode On (sec): " + str(round(totalT)))
                st.subheader("Recirculation State: " + vent)
                if recirc == 1:
                    st.audio('root_dir' + 'recirc.m4a', format='audio/ogg')
                elif recirc == 0:
                    st.audio('root_dir' + 'outside.m4a', format='audio/ogg')
                st.write('')
                st.write('')




if __name__ == "__main__":
    main()







