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

last_publish_time = 0

def classes_info(classes):
    return ', '.join('%s (%.2f)' % pair for pair in classes)

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
    return any([c.kind == kind for c in classes])

def publish_object_detected_message(topic, kind, min_publish_interval):
    global last_publish_time
    now = time.time()
    if now - last_publish_time > min_publish_interval:
        print("Publishing MQTT message")
        last_publish_time  = now

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
    args = parser.parse_args()

    with PiCamera(sensor_mode=4, framerate=10) as camera, \
         CameraPreview(camera, enabled=args.preview), \
         CameraInference(object_detection.model()) as inference:

        current_person_detected_state = False

        # High Quality Stream
        print(f"sending HQ stream to {args.hq_rtsp_url}")
        # HQcmd = "gst-launch-1.0 fdsrc ! h264parse ! rtspclientsink location=rtsp://myuser:mypass@localhost:8554/hqstream debug=false"
        HQcmd = f"gst-launch-1.0 fdsrc ! h264parse ! rtspclientsink location={args.hq_rtsp_url} debug=false"
        HQcmd = shlex.split(HQcmd)
        gstreamerHQ = subprocess.Popen(HQcmd, stdin=subprocess.PIPE)
        camera.start_recording(gstreamerHQ.stdin, splitter_port=1, format='h264', profile='high', intra_period=30, quality=30, sei=True, sps_timing=True)

        if args.enable_lq_stream:
            # Low Quality Stream
            print(f"sending LQ stream to {args.lq_rtsp_url}")
            # LQcmd = "gst-launch-1.0 fdsrc ! h264parse ! rtspclientsink location=rtsp://myuser:mypass@localhost:8554/lqstream debug=false"
            LQcmd = f"gst-launch-1.0 fdsrc ! h264parse ! rtspclientsink location={args.hq_rtsp_url} debug=false"
            LQcmd = shlex.split(LQcmd)
            gstreamerLQ = subprocess.Popen(LQcmd, stdin=subprocess.PIPE)
            camera.start_recording(gstreamerLQ.stdin, splitter_port=2, format='h264', profile='high', intra_period=30, quality=30, sei=True, sps_timing=True, resize=(640, 480))
        
        for result in inference.run(args.num_frames):
            #camera.wait_recording(timeout=1, splitter_port=1)
            #camera.wait_recording(timeout=1, splitter_port=2)
            #classes = image_classification.get_classes(result, top_k=args.num_objects)
            classes = object_detection.get_objects(result, 0.3)
            if classes is not None:
                # print(classes)
                # print(classes_info(classes))
                # camera.annotate_text = '%s (%.2f)' % classes[0]
                # print(str(classes[0]))
                # for object_class in classes:
                #     print(object_detection.Object._LABELS[object_class.kind])
                person_detected = object_kind_detected(classes, object_detection.Object.PERSON)
                # update person_detected flag if different from previous value
                if current_person_detected_state != person_detected:
                    print(f'current_person_detected_state changing from {current_person_detected_state} to {person_detected}')
                    current_person_detected_state = person_detected
                    publish_object_detected_message('', object_detection.Object.PERSON, args.min_mqtt_publish_interval)
                            
if __name__ == '__main__':
    main()