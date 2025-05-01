import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Baca gambar PNG transparan (dengan alpha channel)
j1_icon = cv2.imread('j1.png', cv2.IMREAD_UNCHANGED)
j2_icon = cv2.imread('j2.png', cv2.IMREAD_UNCHANGED)
j3_icon = cv2.imread('j3.png', cv2.IMREAD_UNCHANGED)

def resize_icon(icon, scale):
    if icon is None:
        return None
    new_size = (int(icon.shape[1] * scale), int(icon.shape[0] * scale))
    return cv2.resize(icon, new_size, interpolation=cv2.INTER_AREA)

# Resize ikon agar tidak terlalu besar
j1_icon = resize_icon(j1_icon, 0.7)
j2_icon = resize_icon(j2_icon, 0.7)
j3_icon = resize_icon(j3_icon, 0.7)

def overlay_icon(frame, icon, position):
    if icon is None:
        return frame

    x, y = position
    h, w = icon.shape[:2]

    # Pastikan posisi dalam batas frame
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame

    roi = frame[y:y+h, x:x+w]
    icon_rgb = icon[:, :, :3]
    icon_alpha = icon[:, :, 3] / 255.0

    # Blending icon ke ROI
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - icon_alpha) + icon_rgb[:, :, c] * icon_alpha

    frame[y:y+h, x:x+w] = roi
    return frame

def is_finger_up(tip_y, dip_y, threshold=10):
    return (tip_y < dip_y - threshold)

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ambil landmark jari
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]

                # Konversi ke koordinat pixel
                h, w = frame.shape[:2]
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                index_dip_y = int(index_dip.y * h)

                middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
                middle_dip_y = int(middle_dip.y * h)

                ring_x, ring_y = int(ring_tip.x * w), int(ring_tip.y * h)
                ring_dip_y = int(ring_dip.y * h)

                # Deteksi jari terangkat
                if is_finger_up(index_y, index_dip_y) and j1_icon is not None:
                    icon_pos = (index_x - j1_icon.shape[1] // 2, index_y - j1_icon.shape[0] - 10)
                    frame = overlay_icon(frame, j1_icon, icon_pos)

                if is_finger_up(middle_y, middle_dip_y) and j2_icon is not None:
                    icon_pos = (middle_x - j2_icon.shape[1] // 2, middle_y - j2_icon.shape[0] - 10)
                    frame = overlay_icon(frame, j2_icon, icon_pos)

                if is_finger_up(ring_y, ring_dip_y) and j3_icon is not None:
                    icon_pos = (ring_x - j3_icon.shape[1] // 2, ring_y - j3_icon.shape[0] - 10)
                    frame = overlay_icon(frame, j3_icon, icon_pos)

        cv2.imshow('Finger Icons', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
