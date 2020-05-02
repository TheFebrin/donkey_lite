import pygame
import time
import sys
sys.path.append('/home/pi/Desktop/donkey_lite/')
import car_config
import parts


# init car modules
my_car = car_config.my_car()
bluepill = parts.BluePill(**car_config.bluepill_configs[my_car])
timer = parts.Timer(frequency=20)

# init screen and font
pygame.init()
pygame.font.init()

# set up screen
scr_size = (500, 500)
myfont = pygame.font.SysFont('Comic Sans MS', sum(scr_size) // 30)
window = pygame.display.set_mode(scr_size)
pygame.display.set_caption('Car')
keyboardImg = pygame.image.load('keyboard.jpg')
keyboardImg = pygame.transform.scale(keyboardImg, scr_size)
window.blit(keyboardImg, (0, 0))

# load all keyboard images
keyboardImgW = pygame.image.load('keyboardW.jpg')
keyboardImgW = pygame.transform.scale(keyboardImgW, scr_size)
keyboardImgA = pygame.image.load('keyboardA.jpg')
keyboardImgA = pygame.transform.scale(keyboardImgA, scr_size)
keyboardImgS = pygame.image.load('keyboardS.jpg')
keyboardImgS = pygame.transform.scale(keyboardImgS, scr_size)
keyboardImgD = pygame.image.load('keyboardD.jpg')
keyboardImgD = pygame.transform.scale(keyboardImgD, scr_size)
keyboardImgWA = pygame.image.load('keyboardWA.jpg')
keyboardImgWA = pygame.transform.scale(keyboardImgWA, scr_size)
keyboardImgWD = pygame.image.load('keyboardWD.jpg')
keyboardImgWD = pygame.transform.scale(keyboardImgWD, scr_size)
keyboardImgSA = pygame.image.load('keyboardSA.jpg')
keyboardImgSA = pygame.transform.scale(keyboardImgSA, scr_size)
keyboardImgSD = pygame.image.load('keyboardSD.jpg')
keyboardImgSD = pygame.transform.scale(keyboardImgSD, scr_size)

# set of keys that are held at the moment
# alpha_s = one press change of speed
# alpha_a = one press change of angle
ACTIVE_KEYS = set()
SPEED, ANGLE = 0.25, 0.0
alpha_s, alpha_a = 0.01, 0.05


def process_events(speed, angle):
    global ANGLE
    if 'w' in ACTIVE_KEYS and 'd' in ACTIVE_KEYS:
        ANGLE += alpha_a
        window.blit(keyboardImgWD, (0, 0))
    elif 'w' in ACTIVE_KEYS and 'a' in ACTIVE_KEYS:
        ANGLE -= alpha_a
        window.blit(keyboardImgWA, (0, 0))
    elif 's' in ACTIVE_KEYS and 'd' in ACTIVE_KEYS:
        ANGLE += alpha_a
        window.blit(keyboardImgSD, (0, 0))
    elif 's' in ACTIVE_KEYS and 'a' in ACTIVE_KEYS:
        ANGLE -= alpha_a
        window.blit(keyboardImgSA, (0, 0))
    elif 'w' in ACTIVE_KEYS:
        window.blit(keyboardImgW, (0, 0))
    elif 's' in ACTIVE_KEYS:
        window.blit(keyboardImgS, (0, 0))
    elif 'd' in ACTIVE_KEYS:
        ANGLE += alpha_a
        window.blit(keyboardImgD, (0, 0))
    elif 'a' in ACTIVE_KEYS:
        ANGLE -= alpha_a
        window.blit(keyboardImgA, (0, 0))
    else:
        window.blit(keyboardImg, (0, 0))

    ANGLE = min(ANGLE, 1.0)
    ANGLE = max(ANGLE, -1.0)
    BLACK = (0, 0, 0)
    text = f'SPEED: {speed:.2f} | ANGLE: {angle:.2f}'
    surface = myfont.render(text, False, BLACK)
    window.blit(surface, (0, 0))
    pygame.display.update()


def drive_car():
    if 'w' in ACTIVE_KEYS:
        bluepill.drive(-ANGLE, SPEED)
    elif 's' in ACTIVE_KEYS:
        bluepill.drive(-ANGLE, -SPEED * 1.5)
    else:
        bluepill.drive(-ANGLE, 0)


if __name__ == '__main__':

    print('\n------------------------------------')
    print('Hit ESC or close the window to quit.')
    print('------------------------------------\n')

    print('--------------------------------------')
    print(f'Press P to increase speed by {alpha_s}.')
    print(f'Press O to decrease speed by {alpha_s}')
    print('------------------------------------\n')

    running = True
    while running:
        process_events(SPEED, ANGLE)
        drive_car()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    if 's' not in ACTIVE_KEYS:
                        ACTIVE_KEYS.add('w')
                if event.key == pygame.K_s:
                    if 'w' not in ACTIVE_KEYS:
                        ACTIVE_KEYS.add('s')
                if event.key == pygame.K_a:
                    if 'd' not in ACTIVE_KEYS:
                        ACTIVE_KEYS.add('a')
                if event.key == pygame.K_d:
                    if 'a' not in ACTIVE_KEYS:
                        ACTIVE_KEYS.add('d')
                if event.key == pygame.K_p:
                    SPEED += alpha_s
                if event.key == pygame.K_o:
                    SPEED -= alpha_s
                    SPEED = max(SPEED, 0)
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    if 'w' in ACTIVE_KEYS:
                        ACTIVE_KEYS.remove('w')
                if event.key == pygame.K_s:
                    if 's' in ACTIVE_KEYS:
                        ACTIVE_KEYS.remove('s')
                if event.key == pygame.K_a:
                    if 'a' in ACTIVE_KEYS:
                        ACTIVE_KEYS.remove('a')
                if event.key == pygame.K_d:
                    if 'd' in ACTIVE_KEYS:
                        ACTIVE_KEYS.remove('d')
