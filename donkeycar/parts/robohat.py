#!/usr/bin/env python3
"""
Scripts for operating the RoboHAT MM1 by Robotics Masters with the Donkeycar

author: @wallarug (Cian Byrne) 2019
contrib: @peterpanstechland 2019
contrib: @sctse999 2020

Note: To be used with code.py bundled in this repo. See donkeycar/contrib/robohat/code.py
"""

import time
import logging
import donkeycar as dk

try:
    import serial
except ImportError:
    print("PySerial not found.  Please install: pip install pyserial")

logger = logging.getLogger(__name__)

class RoboHATController:
    '''
    Driver to read signals from SAMD51 and convert into steering and throttle outputs
    Input input signal range: 1000 to 2000
    Output range: -1.00 to 1.00
    '''

    def __init__(self, cfg, debug=False):
        # standard variables
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.recording_latch = None
        self.auto_record_on_throttle = cfg.AUTO_RECORD_ON_THROTTLE
        self.SHOW_STEERING_VALUE = cfg.MM1_SHOW_STEERING_VALUE
        self.DEAD_ZONE = cfg.JOYSTICK_DEADZONE
        self.debug = debug

        self.STOPPED_PWM = cfg.MM1_STOPPED_PWM

        # Forward Enums
        self.MIN_CAR_FORWARD = cfg.MM1_MIN_FORWARD
        self.MAX_CAR_FORWARD = cfg.MM1_MAX_FORWARD

        self.MIN_THROTTLE_FORWARD = cfg.MM1_MIN_THROTTLE_FORWARD
        self.MAX_THROTTLE_FORWARD = cfg.MM1_MAX_THROTTLE_FORWARD

        # Reverse Enums
        self.MIN_CAR_REVERSE = cfg.MM1_MIN_REVERSE
        self.MAX_CAR_REVERSE = cfg.MM1_MAX_REVERSE

        self.MIN_THROTTLE_REVERSE = cfg.MM1_MIN_THROTTLE_REVERSE
        self.MAX_THROTTLE_REVERSE = cfg.MM1_MAX_THROTTLE_REVERSE

        # Steering Enums
        self.STEERING_MID = cfg.MM1_STEERING_MID

        self.prev_throttle_pwm = -1

        try:
            self.serial = serial.Serial(cfg.MM1_SERIAL_PORT, 115200, timeout=1)
        except serial.SerialException:
            print("Serial port not found!  Please enable: sudo raspi-config")
        except serial.SerialTimeoutException:
            print("Serial connection timed out!")

    def shutdown(self):
        try:
            self.serial.close()
        except:
            pass

    def read_serial(self):
        '''
        Read the rc controller value from serial port. Map the value into
        steering and throttle

        Format ####,#### whereas the 1st number is steering and 2nd is throttle
        '''
        line = str(self.serial.readline().decode()).strip('\n').strip('\r')

        output = line.split(", ")
        if len(output) == 2:
            if self.SHOW_STEERING_VALUE:
                print("MM1: steering={}".format(output[0]))

            if output[0].isnumeric() and output[1].isnumeric():
                angle_pwm = float(output[0])
                throttle_pwm = float(output[1])

                if throttle_pwm == self.STOPPED_PWM:
                    # throttle value is in the dead zone - [1475, 1525]
                    self.throttle = 0
                    scaled_throttle_pwm = 0
                elif throttle_pwm > self.STOPPED_PWM:
                    # 1500 - 1525 is dead in the throttle, so [1525, 1625] is our usable throttle range
                    # Car starts moving at 1550, so scale the input pwm from
                    # [1525, 1625] -> [1550, 1555]
                    scaled_throttle_pwm = dk.utils.map_range_float(throttle_pwm,
                                                                   self.MIN_THROTTLE_FORWARD, self.MAX_THROTTLE_FORWARD,
                                                                   self.MIN_CAR_FORWARD, self.MAX_CAR_FORWARD)

                    # Scale from [1550, 1555] -> [0, 1]
                    self.throttle = dk.utils.map_range_float(scaled_throttle_pwm,
                                                             self.MIN_CAR_FORWARD, self.MAX_CAR_FORWARD,
                                                             0, 1.0)

                else:
                    # 1475 - 1500 is dead in the throttle, so [1375, 1475] is our usable throttle range
                    # Car starts moving at 1440, so scale the input pwm from
                    # [1375, 1475] -> [1435, 1440]
                    scaled_throttle_pwm = dk.utils.map_range_float(throttle_pwm,
                                                                   self.MAX_THROTTLE_REVERSE, self.MIN_THROTTLE_REVERSE,
                                                                   self.MAX_CAR_REVERSE, self.MIN_CAR_REVERSE)

                    # Scale from [1435, 1440] -> [-1, 0]
                    self.throttle = dk.utils.map_range_float(scaled_throttle_pwm,
                                                             self.MAX_CAR_REVERSE, self.MIN_CAR_REVERSE,
                                                             -1.0, 0)

                if self.debug:
                    if throttle_pwm != self.prev_throttle_pwm:
                        print(f"throttle_pwm={throttle_pwm}, scaled_throttle={scaled_throttle_pwm}, throttle={self.throttle}")
                        self.prev_throttle_pwm = throttle_pwm

                if angle_pwm >= self.STEERING_MID:
                    # Turn Left
                    self.angle = dk.utils.map_range_float(angle_pwm,
                                                          2000, self.STEERING_MID,
                                                          -1, 0)
                else:
                    # Turn Right
                    self.angle = dk.utils.map_range_float(angle_pwm,
                                                          self.STEERING_MID, 1000,
                                                          0, 1)

                if self.auto_record_on_throttle:
                    was_recording = self.recording
                    self.recording = self.throttle > self.DEAD_ZONE
                    if was_recording != self.recording:
                        self.recording_latch = self.recording
                        logger.debug(f"JoystickController::on_throttle_changes() setting recording = {self.recording}")

                time.sleep(0.01)

    def update(self):
        # delay on startup to avoid crashing
        print("Warming serial port...")
        time.sleep(3)

        while True:
            try:
                self.read_serial()
            except:
                print("MM1: Error reading serial input!")
                break

    def run(self, img_arr=None, mode=None, recording=None):
        """
        :param img_arr: current camera image or None
        :param mode: default user/mode
        :param recording: default recording mode
        """
        return self.run_threaded(img_arr, mode, recording)

    def run_threaded(self, img_arr=None, mode=None, recording=None):
        """
        :param img_arr: current camera image
        :param mode: default user/mode
        :param recording: default recording mode
        """
        self.img_arr = img_arr

        #
        # enforce defaults if they are not none.
        #
        #
        # enforce defaults if they are not none.
        #
        if mode is not None:
            self.mode = mode
        if recording is not None and recording != self.recording:
            logger.debug(f"RoboHATController::run_threaded() setting recording from default = {recording}")
            self.recording = recording
        if self.recording_latch is not None:
            logger.debug(f"RoboHATController::run_threaded() setting recording from latch = {self.recording_latch}")
            self.recording = self.recording_latch
            self.recording_latch = None

        return self.angle, self.throttle, self.mode, self.recording


class RoboHATDriver:
    """
    PWM motor controller using Robo HAT MM1 boards.
    This is developed by Robotics Masters
    """

    def __init__(self, cfg, serial_port=None, debug=False):
        # Initialise the Robo HAT using the serial port
        # Use shared serial port if provided to avoid opening the same port twice
        if serial_port is not None:
            self.pwm = serial_port
        else:
            self.pwm = serial.Serial(cfg.MM1_SERIAL_PORT, 115200, timeout=1)

        self.STOPPED_PWM = cfg.MM1_STOPPED_PWM

        self.MIN_FORWARD = cfg.MM1_MIN_FORWARD
        self.MAX_FORWARD = cfg.MM1_MAX_FORWARD

        self.MIN_REVERSE = cfg.MM1_MIN_REVERSE
        self.MAX_REVERSE = cfg.MM1_MAX_REVERSE

        self.STEERING_MID = cfg.MM1_STEERING_MID
        self.debug = debug

    """
    Steering and throttle should range between -1.0 to 1.0. This function will
    trim value great than 1.0 or less than 1.0
    """

    def trim_out_of_bound_value(self, value):
        if value > 1:
            # print("MM1: Warning, value out of bound. Value = {}".format(value))
            return 1.0
        elif value < -1:
            # print("MM1: Warning, value out of bound. Value = {}".format(value))
            return -1.0
        else:
            return value

    def set_pulse(self, steering, throttle):
        try:
            steering = self.trim_out_of_bound_value(steering)
            throttle = self.trim_out_of_bound_value(throttle)

            if throttle == 0:
                output_throttle = self.STOPPED_PWM

            elif throttle > 0:
                output_throttle = dk.utils.map_range(throttle,
                                                     0, 1.0,
                                                     self.MIN_FORWARD, self.MAX_FORWARD)
            else:
                output_throttle = dk.utils.map_range(throttle,
                                                     -1.0, 0,
                                                     self.MAX_REVERSE, self.MIN_REVERSE)

            if steering > 0:
                output_steering = dk.utils.map_range(steering,
                                                     0, 1.0,
                                                     self.STEERING_MID, 1000)
            else:
                output_steering = dk.utils.map_range(steering,
                                                     -1.0, 0,
                                                     2000, self.STEERING_MID)

            if self.is_valid_pwm_value(output_steering) and self.is_valid_pwm_value(output_throttle):
                if self.debug:
                    print("output_steering=%d, output_throttle=%d" % (output_steering, output_throttle))
                self.write_pwm(output_steering, output_throttle)
            else:
                print(f"Warning: steering = {output_steering}, STEERING_MID = {self.STEERING_MID}")
                print(f"Warning: throttle = {output_throttle}, MAX_FORWARD = {self.MAX_FORWARD}, STOPPED_PWM = {self.STOPPED_PWM}, MAX_REVERSE = {self.MAX_REVERSE}")
                print("Not sending PWM value to MM1")

        except OSError as err:
            print(
                "Unexpected issue setting PWM (check wires to motor board): {0}".format(err))

    def is_valid_pwm_value(self, value):
        if 1000 <= value <= 2000:
            return True
        else:
            return False

    def write_pwm(self, steering, throttle):
        self.pwm.write(b"%d, %d\r" % (steering, throttle))

    def run(self, steering, throttle):
        self.set_pulse(steering, throttle)

    def shutdown(self):
        try:
            self.serial.close()
        except:
            pass
