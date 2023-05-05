import http.server
import json
import os
from datetime import datetime

import numpy as np
import requests
from requests.exceptions import RequestException


class MockEC2MetadataRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/latest/meta-data/spot/instance-action":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                "action": "stop",
                "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()


def run(
    server_class=http.server.HTTPServer,
    handler_class=MockEC2MetadataRequestHandler,
    port=8000,
):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting mock EC2 metadata server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()


def is_spot_instance_terminating():
    termination_notice_url = (
        "http://169.254.169.254/latest/meta-data/spot/instance-action"
    )

    try:
        response = requests.get(termination_notice_url, timeout=1)
    except RequestException:
        return False

    if response.status_code == 200:
        print("Spot Instance termination notice received")
        return True

    return False


def detect_ec2_interruption():
    rand_int = np.random.randint(0, 1000)
    if rand_int >= 1000:
        print("Interrupting EC2")
        return True
    else:
        return False


class EC2Interruption(Exception):
    pass


def touch_empty_file(filename):
    with open(filename, "a"):
        os.utime(filename, None)


def remove_interrupted_file():
    try:
        os.remove("interrupted")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    run()
