import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Setup pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Open webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
p_time = 0
c_time = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((id, cx, cy))

        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            x1, y1 = lm_list[4][1], lm_list[4][2]   # Thumb tip
            x2, y2 = lm_list[8][1], lm_list[8][2]   # Index finger tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Hand range 20 - 200
            # Volume range min_vol to max_vol
            vol = np.interp(length, [20, 200], [min_vol, max_vol])
            vol_bar = np.interp(length, [20, 200], [400, 150])
            vol_per = np.interp(length, [20, 200], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            # Draw volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0,255,0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0,255,0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_per)} %', (40, 430), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 3)

    # Calculate and show FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) != 0 else 0
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 3)

    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
