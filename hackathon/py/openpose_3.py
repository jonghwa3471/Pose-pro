#다리를 올렸다 내리는 재활 운동

import cv2
import numpy as np
import time
from flask import Flask, Response, render_template

# 카메라 해상도
width = 600
height = 400

#오픈포즈에서 기본 제공되는 부위별 값
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

#오픈포즈에서 기본 제공되는 선으로 연결될 부위
POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 path
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

cap2 = cv2.VideoCapture(0)  # 웹캠 인덱스 (0부터 시작) 설정

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

app = Flask(__name__)


def generate_virtual_frame():

    # 웹캠으로부터 프레임 읽기
    ret, frame = cap2.read()

    # 높이,너비 및 채널 수를 가져옴
    imageHeight, imageWidth, _ = frame.shape

    # 이미지를 전처리하여 신경망에 입력할 수 있는 형태로 변경
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    #신경막의 입력을 설정
    net.setInput(inpBlob)

    #신경망의 예측 결과를 반환
    output = net.forward()

    # 출력의 높이와 너비를 가져옴
    H = output.shape[2]
    W = output.shape[3]

    points = []

    for i in range(0, 15):
        # 해당 신체부위 신뢰도 얻기
        probMap = output[0, i, :, :]

        # global 최대값 찾기
        _, prob, _, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = int(imageWidth * point[0] / W)
        y = int(imageHeight * point[1] / H)

        # 키포인트 검출한 결과가 0.1보다 크면(검출한 곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > 0.1:
            points.append((x, y))
        else:
            points.append(None)

    partA = BODY_PARTS["RKnee"]   #오른무릎
    partB = BODY_PARTS["Chest"]   #가슴
    partC = BODY_PARTS["LKnee"]   #왼쪽무릎

    #부위 연결
    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

    if points[partB] and points[partC]:
        cv2.line(frame, points[partB], points[partC], (0, 255, 0), 2)

    if points[partC] != None:
        if isinstance(points[partA], tuple) and isinstance(points[partB], tuple):
            x1, y1 = points[partA] # 오른무릎 좌표
            x2, y2 = points[partB] # 가슴 좌표
            x3, y3 = points[partC] # 왼쪽무릎 좌표

            # 좌표 지점에 원을 그려서 표시
            cv2.circle(frame, (x1, y1), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, (x2, y2), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, (x3, y3), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

            #각도 계산
            angle = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
            angle = np.degrees(angle)

            #각도가 0 미만이면 360을 더해서 각도를 양수로 수정
            if angle < 0:
                angle += 360

            #각도가 180 초과라면 360을 빼서 각도를 수정한뒤 -1을 곱해 양수로 변환
            if angle > 180:
                angle -= 360
                angle = angle * -1

            # 각도가 30미만 60초과라면 잘못된 자세라는 판정으로 선을 빨간색으로 변경
            if angle > 60: 
                cv2.line(frame, points[partA], points[partB], (0, 0, 255), 2)
                cv2.line(frame, points[partB], points[partC], (0, 0, 255), 2)
            if angle < 30: 
                cv2.line(frame, points[partA], points[partB], (0, 0, 255), 2)
                cv2.line(frame, points[partB], points[partC], (0, 0, 255), 2)

    # 캡처된 프레임 반환
    return frame


def capture_frame():

    # 가상의 프레임 생성
    frame = generate_virtual_frame()

    # 프레임을 바이트 스트림으로 변환
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return frame_bytes

def generate_frames():

    #현재 시간을 삽입
    last_frame_time = time.time()

    while True:
        # 비디오 프레임 생성 로직
        frame = capture_frame()

        current_time = time.time()

        if current_time - last_frame_time < 0.01:  # 0.01초 (프레임 간격 조정을 위한 임의의 값)
            continue

        # 프레임 반환   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        last_frame_time = current_time

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port = 5502, debug=True)