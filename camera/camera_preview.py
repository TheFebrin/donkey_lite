from picamera import PiCamera
from time import sleep

sec = int(input('How many seconds of preview?: '))
alpha = int(input('Transparency, (0 - 255): '))

camera = PiCamera()
camera.rotation = 180
# camera.resolution = (600, 720)
camera.start_preview(alpha=alpha)
sleep(sec)
camera.stop_preview()
