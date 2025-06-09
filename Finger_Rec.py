import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

 
    if hand_label == 'Right':
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    for tip_id in tips_ids[1:]:
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[tip_id - 2].y
        mcp_y = hand_landmarks.landmark[tip_id - 3].y

        if tip_y < pip_y and tip_y < mcp_y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                finger_count = count_fingers(hand_landmarks, label)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                coords = hand_landmarks.landmark[0]
                h, w, _ = image.shape
                cx, cy = int(coords.x * w), int(coords.y * h)
                cv2.putText(image, f'{label}: {finger_count}', (cx - 30, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


        cv2.imshow('Finger Counter', image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
