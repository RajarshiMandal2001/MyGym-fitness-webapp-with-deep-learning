from django.shortcuts import render, HttpResponse
# Create your views here.
def index(request):
    # return HttpResponse('This is home page')
    return render(request, 'templetes/input.html', {'reg_output': 23})


def about(request):
    sending_variable = {
        'value_got_as_prop_1': 'HAPPY Coding',
        'value_got_as_prop_2': 'HAPPY django',
    }
    return render(request, 'templetes/test.html', sending_variable)

def regressor(request):
    x = request.GET.get('q', '')
    print("x=", x)
    print("ndnkd")
    return HttpResponse(f"value is {call_regression(float(x))}")

def regressor_from_input(request):
    return render(request, "templetes/regression.html")

def get_video(request):
    message = "Hello, this is a plain text message! coming from Django server"
    return HttpResponse(message, content_type="text/plain")

def get_video_classification(request):
    print("function invoked!")
    import numpy as np 
    import cv2     # for capturing videos
    import math   # for mathematical operations
    import pandas as pd
    from keras.preprocessing import image   # for preprocessing the images
    import numpy as np    # for mathematical operations
    from keras.utils import np_utils
    # from skimage.transform import resize   # for resizing images
    from glob import glob

    from keras.models import Sequential
    from keras.applications.vgg16 import VGG16
    from keras.layers import Dense, Dropout
    import keras.utils as image
    import numpy as np
    import statistics as s

    cap = cv2.VideoCapture('C:/Users/Rajarshi/Downloads/FetchedVideo.mp4')   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    count, x = 0, 1  
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename ='E:/my_python programs/ExerciseVid/videoClassifierTrainedModelFrames/'+'frame_'+ str(count) + '.jpg'
            cv2.imwrite(filename, frame)
            count+=1
    cap.release()


    base_model = VGG16(weights='imagenet', include_top=False)
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(25088,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='softmax'))

    model.load_weights("E:/my_python programs/ExerciseVid/weight.hdf5")
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    images = glob("E:/my_python programs/ExerciseVid/videoClassifierTrainedModelFrames/*.jpg")
    prediction_images = []
    predict = []
    y = ['Bench press', 'Biceps curl', 'Chest fly machine', 'Deadlift', 'decline bench press', 'hip thrust', 'Incline bench press', 'Lat pulldown', 'leg extension', 'leg raises', 'plank', 'Pull Up', 'Push-up', 'russian twist', 'shoulder press', 'tricep dips', 'Tricep Pushdown']
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        prediction_images.append(img)

    prediction_images = np.array(prediction_images)
    prediction_images = base_model.predict(prediction_images)
    print("extracted features:", prediction_images)
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
    prediction = np.argmax(model.predict(prediction_images),axis=1)
    print("per frame prediction",prediction)
    predict.append(y[s.mode(prediction)])
    print("predicted exercise:",predict)

    message = f"Hello, You may be practicing {predict}"
    return HttpResponse(message, content_type="text/plain")

def input(request):
    # from django.shortcuts import redirect
    # import numpy as np
    # response = redirect('/about')
    # return response
    # user_data = request.GET.dict()
    # username = login_data.get("username")
    print("check....................................................................")
    print(request.GET)
    print(request.POST)
    user_data = request.GET.get('fname')
    prediction = call_regression(float(user_data))
    return render(request, 'templetes/input.html', {'reg_output': prediction})

    # if request.GET:
    #     user_data = request.GET.get('fname')
    # # int_features=[int(x) for x in request.form.values()]
    # # final=[np.array(int_features)]
    #     prediction = call_regression(float(user_data))
    #     return render('templetes/input.html', {'reg_output': prediction})
    # else:
    #     return HttpResponse('Nothing to show bro')







import pandas as pd
from scipy import stats

def call_regression(num):
    data = pd.DataFrame({'Exp':[1.1, 1.3, 1.5, 2.0, 2.2, 2.8],'salary':[39,46,37,43,40,45]})
    x = data['Exp']
    y = data['salary']

    def predict(B0, B1, new_x):
        y = B0 + B1 * new_x
        return y

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    # print("imported res",predict(intercept,slope,4.1))

    return predict(intercept,slope, num)