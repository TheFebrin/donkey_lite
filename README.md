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

## Connection
#### To set up Raspberry Pi look [here](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up).

#### Connect Raspberry Pi to WiFi.

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

#### Connect your PC to Raspberry Pi using SSH.

<details><summary><b>Show instructions</b></summary>

  1. Make sure SSH and VNC is enabled. If not go to `sudo raspi-config` then `Interfacing Options`.

  2. Get your Raspberry Pi ip adress. `ifconfig`

  3. Connect your PC to the same WiFi network.

  4. Connect your PC to Raspberry using for e.g. [PuTTY](https://www.putty.org/).

</details>


## Miscellaneous
To build the CPP module run `python setup.py build_ext --inplace`.

## Screenshots
TODO

## Technologies
* Python - version 3.7.3

## Status
Project is: _just_started_,

## Inspiration
Most of the code is adapted from https://github.com/autorope/donkeycar
