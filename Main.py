from mylib.cnn import *
from mylib.pre_processing import *


# Define video argument.
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", required = True, help = "path to the video file")
args = vars(parser.parse_args())

# read video file, can be camera also.
cap = cv2.VideoCapture(args['video'])
while (cap.isOpened()):

    plateRet, plateFrame = cap.read()

    # Experiments
    #plate_Original, plate_morphEx, edge = preprocess(plateFrame, (42,10), True)
    #plate_Original, plate_morphEx, edge = preprocess(plateFrame, (34,8), False)
    plate_Original, plate_morphEx, edge = preprocess(plateFrame, (15,3), True)

    # find plate countours
    plate_countours,_ = cv2.findContours(plate_morphEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for plt_countour in plate_countours:

        # ratio of width to hieght
        aspect_ratio_range, area_range = (2.8, 8), (1000, 18000)
        #aspect_ratio_range, area_range = (2.8, 3), (1000, 18000)

        # validate the countours, return boolean (True, False)
        if contour_vladition(plt_countour, plate_morphEx, aspect_ratio_range, area_range):
            rect = cv2.minAreaRect(plt_countour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(plate_Original, [box], 0, (0,255,0), 1) #change position after CNN
            x_s, y_s = [i[0] for i in box], [i[1] for i in box]
            x1, y1 = min(x_s), min(y_s)
            x2, y2 = max(x_s), max(y_s)

            angle = rect[2]
            if angle < -45: angle += 90

            W, H = rect[1][0], rect[1][1]
            aspect_ratio = float(W)/H if W > H else float(H)/W

            center = ((x1+x2)/2, (y1+y2)/2)
            size = (x2-x1, y2-y1)
            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
            tmp = cv2.getRectSubPix(edge, size, center)
            Tmp_w = H if H > W else W
            Tmp_h = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(Tmp_w),int(Tmp_h)), (size[0]/2, size[1]/2))
            __,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            white_pixels = 0
            for x in range(tmp.shape[0]):
                for y in range(tmp.shape[1]):
                    if tmp[x][y] == 255:
                        white_pixels += 1

            edge_density = float(white_pixels)/(tmp.shape[0]*tmp.shape[1])

            tmp = cv2.getRectSubPix(plateFrame, size, center)
            tmp = cv2.warpAffine(tmp, M, size)
            Tmp_w = H if H > W else W
            Tmp_h = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(Tmp_w),int(Tmp_h)), (size[0]/2, size[1]/2))
            cv2.imshow("plate_original", plate_Original)


            tmp = im_reshape(tmp, plate_img_size, "plate_buffer.jpg")

            data = tmp.reshape(plate_img_size, plate_img_size, 1)
            plate_model_out = plate_model.predict([data])[0]
            if not np.argmax(plate_model_out) == 1:
                cv2.drawContours(plate_Original, [box], 0, (0,0,255), 2) #change position after CNN
                charOrigin = copy.copy(tmp)
                charGaussian = cv2.GaussianBlur(tmp, (3,3), 0)
                charThresh = cv2.adaptiveThreshold(charGaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)

                x, y, w, h = 0, 0, 0, 0
                charsBuffer = []

                charContours,_ = cv2.findContours(charThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for charContour in charContours:
                    area = cv2.contourArea(charContour)
                    if area > 200 and area < 800:   [x,y,w,h] = cv2.boundingRect(charContour)
                    if h > 25 and h < 75 and w > 10 and w < 45:
                        if not len(charOrigin[y:y+h, x:x+w]) < 10:
                            Buffer = copy.copy(tmp[y:y+h, x:x+w])
                            cv2.rectangle(charOrigin, (x,y), (x+w,y+h), (255,0,0), 1)
                            charsBuffer.append([x, Buffer])

                charsBuffer = sorted(charsBuffer, key= lambda x: x[0])
                TrueChars = []

                for i in range(len(charsBuffer)):
                    if i == 0:  TrueChars.append(charsBuffer[i])
                    elif charsBuffer[i][0] != charsBuffer[i-1][0]: TrueChars.append(charsBuffer[i])

                del (charsBuffer)
                if len(TrueChars) >= 4 or len(TrueChars) <= 10:
                    cv2.imshow("Plate ", charOrigin)

                    string_buffer = []
                    for i in range(len(TrueChars)):
                        Buffer = TrueChars[i][1]
                        Buffer = im_reshape(Buffer, char_img_size, "char_buffer.jpg")
                        data1 = Buffer.reshape(char_img_size, char_img_size, 1)
                        char_model_out = char_model.predict([data1])[0]
                        if not np.argmax(char_model_out) == 36:
                            string_buffer.append(code_to_char(char_model_out))
                            pass
                        pass
                    if len(string_buffer) >=7 and len(string_buffer) <= 8:
                        t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                        print("[{x}] ===> [{y}]".format(x=t, y=string_buffer))
                        pass
                    pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

os.system("rm char_buffer.jpg")
os.system("rm plate_buffer.jpg")
