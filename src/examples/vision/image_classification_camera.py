#!/usr/bin/env python3
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Camera image classification demo code.

Runs continuous image classification on camera frames and prints detected object
classes.

Example:
image_classification_camera.py --num_frames 10
"""
import argparse
import contextlib

from aiy.vision.inference import CameraInference
#from aiy.vision.models import image_classification
from aiy.vision.models import object_detection
from picamera import PiCamera

import time
import shlex, subprocess, os
import paho.mqtt.client as mqtt
import json

class ObjectEventPublisher():
    def __init__(self, mqtt_client_name, mqtt_topic, min_publish_interval) -> None:
        self.__mqtt_server_connected = False
        self.__last_publish_time = 0
        self.__client = mqtt.Client(mqtt_client_name)
        self.__min_publish_interval = min_publish_interval
        self.__mqtt_topic = mqtt_topic

    def __on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT server {client} {userdata} {flags} {rc}")
        self.__mqtt_server_connected = True

    def __on_connect_fail(self, ):
        print("Failed to connect to MQTT server")
        self.__mqtt_server_connected = False

    def connect(self, mqtt_host, mqtt_port, mqtt_username, mqtt_password):
        self.__client.on_connect = self.__on_connect
        self.__client.on_connect_fail = self.__on_connect_fail
        self.__client.username_pw_set(mqtt_username, mqtt_password)
        self.__client.connect(mqtt_host, port=mqtt_port)
        self.__client.loop_start()

    def publish(self, kind, detected):
        if not self.__mqtt_server_connected:
            print("Not connected to MQTT server")
            return

        now = time.time()
        if self.__mqtt_server_connected and now - self.__last_publish_time > self.__min_publish_interval:
            payload = json.dumps({ "classification": kind, "detected": detected })
            print(f"Publishing {payload}")
            # self.__client.publish(self.__mqtt_topic, json.dumps({ "classification": kind, "detected": "On" if detected else "Off" }))
            self.__client.publish(self.__mqtt_topic, payload)
            self.__last_publish_time = now

def classes_info(classes):
    return ', '.join('%s (%.2f)' % pair for pair in classes)

def on_mqtt_connect(client, userdata, flags, rc):
    print(f"MQTT client connected {client} {userdata} {flags} {rc}")

def on_mqtt_connect_fail():
    print("MQTT connection failed")

@contextlib.contextmanager
def CameraPreview(camera, enabled):
    if enabled:
        camera.start_preview()
    try:
        yield
    finally:
        if enabled:
            camera.stop_preview()

def object_kind_detected(classes, kind):
    return classes and any([c.kind == kind for c in classes])

def main():
    parser = argparse.ArgumentParser('Image classification camera inference example.')
    parser.add_argument('--num_frames', '-n', type=int, default=None,
        help='Sets the number of frames to run for, otherwise runs forever.')
    parser.add_argument('--num_objects', '-c', type=int, default=3,
        help='Sets the number of object interences to print.')
    parser.add_argument('--nopreview', dest='preview', action='store_false', default=True,
        help='Enable camera preview')
    parser.add_argument('--hq_rtsp_url', default='rtsp://myuser:mypass@localhost:8554/hqstream',
        help='High Qualify RTSP URL')
    parser.add_argument('--lq_rtsp_url', default='rtsp://myuser:mypass@localhost:8554/lqstream',
        help='Low Qualify RTSP URL')
    parser.add_argument('--enable_lq_stream', default=False)
    parser.add_argument('--min_mqtt_publish_interval', default=30,
        help='Min. interval between publishing MQTT message for detected object')
    parser.add_argument('--mqtt_topic', default='ipcam/aiy-vision/object-detected',
        help='MQTT topic for publishing object detected events')
    parser.add_argument('--mqtt_client_name', default='aiy-vision',
        help='MQTT client name')
    parser.add_argument('--mqtt_server_host', default='localhost',
        help='MQTT server hostname')
    parser.add_argument('--mqtt_server_port', default=1883,
        help='MQTT server port')
    parser.add_argument('--mqtt_username', default='user', help='MQTT username')
    parser.add_argument('--mqtt_password', default='password', help='MQTT password')
    args = parser.parse_args()

    object_event_publisher = ObjectEventPublisher(args.mqtt_client_name, args.mqtt_topic, args.min_mqtt_publish_interval)
    object_event_publisher.connect(args.mqtt_server_host, args.mqtt_server_port, args.mqtt_username, args.mqtt_password)

    with PiCamera(sensor_mode=4, framerate=10) as camera, \
         CameraPreview(camera, enabled=args.preview), \
         CameraInference(object_detection.model()) as inference:

        # High Quality Stream
        print(f"sending HQ stream to {args.hq_rtsp_url}")
        HQcmd = f"gst-launch-1.0 fdsrc ! h264parse ! rtspclientsink location={args.hq_rtsp_url} debug=false"
        HQcmd = shlex.split(HQcmd)
        gstreamerHQ = subprocess.Popen(HQcmd, stdin=subprocess.PIPE)
        camera.start_recording(gstreamerHQ.stdin, splitter_port=1, format='h264', profile='high', intra_period=30, quality=30, sei=True, sps_timing=True)

        if args.enable_lq_stream:
            # Low Quality Stream
            print(f"sending LQ stream to {args.lq_rtsp_url}")
            LQcmd = f"gst-launch-1.0 fdsrc ! h264parse ! rtspclientsink location={args.hq_rtsp_url} debug=false"
            LQcmd = shlex.split(LQcmd)
            gstreamerLQ = subprocess.Popen(LQcmd, stdin=subprocess.PIPE)
            camera.start_recording(gstreamerLQ.stdin, splitter_port=2, format='h264', profile='high', intra_period=30, quality=30, sei=True, sps_timing=True, resize=(640, 480))
        
        for result in inference.run(args.num_frames):
            #camera.wait_recording(timeout=1, splitter_port=1)
            #camera.wait_recording(timeout=1, splitter_port=2)
            #classes = image_classification.get_classes(result, top_k=args.num_objects)
            classes = object_detection.get_objects(result, 0.3)
            person_detected = object_kind_detected(classes, object_detection.Object.PERSON)
            if person_detected:
                # camera.annotate_text = '%s (%.2f)' % classes[0]
                # for object_class in classes:
                #     print(object_detection.Object._LABELS[object_class.kind])
                # update person_detected flag if different from previous value
                object_event_publisher.publish(object_detection.Object.PERSON, person_detected)
                print(f"Person detected")
            
            time.sleep(1)

if __name__ == '__main__':
    main()