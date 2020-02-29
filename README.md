# Donkey car lite
> Simpler version of donkey self driving car

## Table of contents
* [General info](#general-info)
* [Connection](#connection)
* [Miscellaneous](#miscellaneous)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Status](#status)
* [Inspiration](#inspiration)

## General info
Donkeycar is minimalist and modular self driving library for Python. It is developed for hobbyists and students with a focus on allowing fast experimentation and easy community contributions.
<br>
We use Raspberry Pi 3 and the STM32 board a.k.a Blue Pill to communicate with car.

## Setup Remote Access
#### To set up Raspberry Pi look [here](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up).

#### Connect Raspberry Pi to WiFi

<details><summary><b>Show instructions</b></summary>

1. Check for available WiFi on your Raspberry Pi:

    ```sh
    $ sudo iwlist wlan0 scan |more
    ```

2. Input the ESSID and password into the configuration file.

    ```sh
    $ sudo wpa_passphrase "login" "password"
    ```

    Append it to configuration file.
    ```sh
    $ sudo wpa_passphrase "login" "password" | sudo tee -a /etc/wpa_supplicant/wpa_supplicant.conf
    ```

    Go to `wpa_supplicant.conf` file and delete non hashed password. (`ctr + k` to cut line, then `ctr + x` to save)
    ```sh
    $ sudo nano -w /etc/wpa_supplicant/wpa_supplicant.conf
    ```

3. Reconfigure and connect to a new network.

    To check connection.
    ```sh
    $ ifconfig wlan0
    ```

    To reconfigure connection.
    ```sh
    $ sudo wpa_cli -i wlan0
    ```
    You should get `OK` output.

    Now if you check connection again there will be information about your network.
</details>

#### Connect your PC to Raspberry Pi using SSH

<details><summary><b>Show instructions</b></summary>

  1. Make sure SSH and VNC is enabled. If not go to `sudo raspi-config` then `Interfacing Options`.

  2. Connect your PC to the same WiFi network.

  3. Get your Raspberry Pi ip adress. `ifconfig` or use IP Scanner.

  4. Connect your PC to Raspberry using for e.g. [PuTTY](https://www.putty.org/) (only command line)
     or [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) (full preview).

</details>


#### Setup Remote File Access

<details><summary><b>Show instructions</b></summary>

  1. Install Samba File Server on Raspberry Pi.
      ```sh
      $ sudo apt-get install samba samba-common-bin -y
      ```

      ```sh
      $ sudo rm /etc/samba/smb.conf
      ```

      ```sh
      $ sudo nano /etc/samba/smb.conf
      ```

  2. Paste in the following lines into the nano editor.
      ```sh
      [global]
      netbios name = Pi
      server string = The PiCar File System
      workgroup = WORKGROUP

      [HOMEPI]
      path = /home/pi
      comment = No comment
      browsable = yes
      writable = Yes
      create mask = 0777
      directory mask = 0777
      public = no
      ```

  3. Create samba password.
      ```sh
      $ sudo smbpasswd -a pi
      ```

  4. Restart samba server.
      ```sh
      $ sudo service smbd restart
      ```

  5. Go to your PC (Windows), open `cmd` and type:
      ```sh
      C:\Users\My_user_name>net use r: \\your_raspberry_ip\homepi
      ```
      `dir r:` to check directory <br>
      `r:` to switch to Pi your disk

  From now on you are able to access your Pi's content in `This PC` as it's an external drive.
  <br>

  For Mac users check [this](https://osxdaily.com/2010/09/20/map-a-network-drive-on-a-mac/).
</details>


#### Install USB Camera

<details><summary><b>Show instructions</b></summary>

  1. To enable video preview using VNC, open the VNC Server dialog (top bar on the RPi), navigate to Menu > Options > Troubleshooting, and select `Enable direct capture mode`.

  2. Run `raspivid -f` in terminal to see if it works.

  3. You might also use cheese if you want.
      ```sh
      $ sudo apt-get install cheese -y
      ```
      (It didn't work for me, tho.)

  4. Simple python scipt worked the best for me.
      ```sh
      from picamera import PiCamera
      from time import sleep

      camera = PiCamera()
      camera.rotation = 180
      camera.start_preview(alpha=200)
      sleep(5)
      camera.stop_preview()
      ```

</details>

## Miscellaneous
* To build the CPP module run `python setup.py build_ext --inplace`.

* To check RPi temperature type `vcgencmd measure_temp
`

## Screenshots
TODO

## Technologies
* Python - version 3.7.3
TODO

## Status
Project is: _just_started_,

## Inspiration
Most of the code is adapted from https://github.com/autorope/donkeycar
