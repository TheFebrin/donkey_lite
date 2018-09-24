import datetime
import numpy as np

import car_config
import parts

MY_CAR = '10'
bluepill = parts.BluePill(**car_config.bluepill_configs[MY_CAR])

timer = parts.Timer(frequency=20)
cam = parts.PiCamera()
stream = parts.WebCameraStream()

# add tub to save data
inputs = ['user_angle', 'user_throttle', 'distance', 'image_array', 'timestamp']
types = ['float', 'float', 'float', 'image_array', 'str']

# single tub
tub = parts.TubWriter(path='./recording_car_tub', inputs=inputs, types=types)

is_recording = False

try:
    print("Car loop started. Start driving to record.")
    while True:
        timer.tick()
        timestamp = str(datetime.datetime.utcnow())
        car_status = bluepill.get_status()
        if not is_recording and car_status.user_throttle > 0.1:
            is_recording = True
            print(timestamp, "Recording enabled")
        elif is_recording and car_status.user_throttle < -0.2:
            is_recording = False
            print(timestamp, "Recording disabled")
        img = cam.get_image()
        stream.set_image(img)
        if is_recording:
            tub.write(car_status.user_angle,
                    car_status.user_throttle,
                    car_status.distance,
                    img,
                    timestamp)
finally:
    bluepill.stop_and_disengage_autonomy()
