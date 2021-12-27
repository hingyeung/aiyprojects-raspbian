# RTSP feed for image_classification_camera
This modified version of the original `image_classification_camera.py` publishes camera feed as RTSP stream.

## Steps
1. start up [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server)
    ```bash
    sudo systemctl start rtsp-simple-server
    ```
1. start the `image_classification_camera.py`
    ```bash
    python3 image_classification_camera.py
    ```
1. View the RTSP feed at [rtsp://myuser:mypass@<aiy-vision-hostname>:8554/hqstream](rtsp://myuser:mypass@<aiy-vision-hostname>:8554/hqstream)
    Username and password are set up in `rtsp-simple-server.yml`.