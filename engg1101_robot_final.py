import cv2
import mediapipe as mp
import math
import time
import serial
import signal
import sys

# Bluetooth Serial Config
BT_PORT = "/dev/cu.ENGG1101_K1"
BAUD_RATE = 115200

SEND_INTERVAL = 0.25
MOVE_TIME_MS = 220

PULSE_MIN = 1320
PULSE_MAX = 3000
PULSE_CENTER = 2047

# Robot Arm Channel Config
BASE_SERVO = 0
SHOULDER_SERVO = 1
ELBOW_SERVO = 2
GRIPPER_SERVO = 3

HOME_ANGLES = {
    BASE_SERVO: 90.0,
    SHOULDER_SERVO: 105.0,
    ELBOW_SERVO: 105.0,
    GRIPPER_SERVO: 90.0
}

current_angles = HOME_ANGLES.copy()

ALPHA = 0.45
ANGLE_DEADZONE = 2.0

last_print_time = 0
PRINT_INTERVAL = 0.3
last_send_time = 0
last_sent_pulses = None

cap = None
ser = None

last_hand_seen_time = time.time()
RESET_DELAY = 0.7

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Clean Exit
def cleanup_and_exit(signum=None, frame=None):
    print("\nShutting down cleanly...")

    global ser, cap

    try:
        if ser is not None and ser.is_open:
            ser.close()
            print("Bluetooth serial closed.")
    except Exception as e:
        print("Serial close error:", e)

    try:
        if cap is not None:
            cap.release()
    except Exception as e:
        print("Camera release error:", e)

    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print("OpenCV close error:", e)

    sys.exit(0)


signal.signal(signal.SIGINT, cleanup_and_exit)


# Helper Functions
def clamp(value, low, high):
    return max(low, min(high, value))


def map_range(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def smooth_angle(prev, target, alpha=ALPHA):
    return prev * (1 - alpha) + target * alpha


def distance_3d(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def distance_2d(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def apply_deadzone(prev_angle, new_angle, deadzone=ANGLE_DEADZONE):
    if abs(new_angle - prev_angle) < deadzone:
        return prev_angle
    return new_angle


def is_fist(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    fingertips = [4, 8, 12, 16, 20]
    dists = [distance_2d(wrist, hand_landmarks.landmark[i]) for i in fingertips]
    avg_dist = sum(dists) / len(dists)
    return avg_dist < 0.18


def extract_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    dx = middle_mcp.x - wrist.x
    wrist_y = wrist.y
    palm_size = distance_3d(wrist, middle_mcp)
    pinch_dist = distance_3d(thumb_tip, index_tip)

    return dx, wrist_y, palm_size, pinch_dist


def landmark_to_servo_targets(hand_landmarks):
    dx, wrist_y, palm_size, pinch_dist = extract_features(hand_landmarks)

    base_target     = map_range(dx, -0.08, 0.08, 180, 0)
    shoulder_target = map_range(wrist_y, 0.15, 0.55, 170, 15)
    elbow_target    = map_range(wrist_y, 0.15, 0.55, 10, 170)
    gripper_target  = map_range(pinch_dist, 0.02, 0.22, 150, 30)

    return {
        BASE_SERVO: clamp(base_target, 0, 180),
        SHOULDER_SERVO: clamp(shoulder_target, 0, 100),
        ELBOW_SERVO: clamp(elbow_target, 0, 180),
        GRIPPER_SERVO: clamp(gripper_target, 10, 160),
    }


def maybe_print_debug(current_angles, dx, wrist_y, palm_size, pinch_dist):
    global last_print_time
    now = time.time()
    if now - last_print_time >= PRINT_INTERVAL:
        print(
            f"dx={dx:.3f}, wrist_y={wrist_y:.3f}, palm={palm_size:.3f}, pinch={pinch_dist:.3f} | "
            f"BASE={int(current_angles[BASE_SERVO])}, "
            f"SHOULDER={int(current_angles[SHOULDER_SERVO])}, "
            f"ELBOW={int(current_angles[ELBOW_SERVO])}, "
            f"GRIPPER={int(current_angles[GRIPPER_SERVO])}"
        )
        last_print_time = now


def angle_to_pulse(angle):
    angle = clamp(angle, 0, 180)
    return int(PULSE_MIN + (angle / 180.0) * (PULSE_MAX - PULSE_MIN))


def check_robot_connection(ser):
    ser.reset_input_buffer()
    time.sleep(0.5)

    for attempt in range(1, 4):
        print(f"Checking robot connection... attempt {attempt}")
        ser.write(b"POS\n")
        ser.flush()

        deadline = time.time() + 2.0
        reply = ""

        while time.time() < deadline:
            chunk = ser.read_all().decode(errors="ignore")
            if chunk:
                reply += chunk
                if "POS," in reply:
                    print("Robot reply:", repr(reply))
                    return True
            time.sleep(0.1)

        print("Robot reply:", repr(reply))

    return False


def send_center_position(ser):
    cmd = f"SET,2047,2047,2047,2047,1200\n"
    print("INIT:", cmd.strip())
    ser.write(cmd.encode())
    ser.flush()
    time.sleep(2)


def should_send_command(pulses, threshold=55):
    global last_sent_pulses

    if last_sent_pulses is None:
        last_sent_pulses = pulses
        return True

    changed = any(abs(pulses[i] - last_sent_pulses[i]) >= threshold for i in range(4))

    if changed:
        last_sent_pulses = pulses
        return True

    return False


def send_angles_to_robot(ser, angles):
    global last_send_time

    now = time.time()
    if now - last_send_time < SEND_INTERVAL:
        return

    s1 = angle_to_pulse(angles[BASE_SERVO])
    s2 = angle_to_pulse(angles[SHOULDER_SERVO])
    s3 = angle_to_pulse(angles[ELBOW_SERVO])
    s4 = angle_to_pulse(angles[GRIPPER_SERVO])

    pulses = [s1, s2, s3, s4]

    if not should_send_command(pulses, threshold=35):
        return

    last_send_time = now

    cmd = f"SET,{s1},{s2},{s3},{s4},{MOVE_TIME_MS}\n"
    print("SEND:", cmd.strip())
    ser.write(cmd.encode())
    ser.flush()


# Main
ser = serial.Serial(BT_PORT, BAUD_RATE, timeout=2)
time.sleep(3)

if not check_robot_connection(ser):
    print("Robot is not responding over Bluetooth.")
    cleanup_and_exit()

send_center_position(ser)

# 실제 초기자세와 내부 상태를 맞춰줌
current_angles = HOME_ANGLES.copy()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not be opened.")
    cleanup_and_exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            last_hand_seen_time = time.time()

            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            target_angles = landmark_to_servo_targets(hand_landmarks)

            for ch in target_angles:
                smoothed = smooth_angle(current_angles[ch], target_angles[ch])
                filtered = apply_deadzone(current_angles[ch], smoothed)
                current_angles[ch] = filtered

            send_angles_to_robot(ser, current_angles)

            dx, wrist_y, palm_size, pinch_dist = extract_features(hand_landmarks)
            maybe_print_debug(current_angles, dx, wrist_y, palm_size, pinch_dist)

            cv2.putText(frame, "Mode: Bluetooth Robot Control", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cv2.putText(frame, f"BASE: {int(current_angles[BASE_SERVO])}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"SHOULDER: {int(current_angles[SHOULDER_SERVO])}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"ELBOW: {int(current_angles[ELBOW_SERVO])}", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"GRIPPER: {int(current_angles[GRIPPER_SERVO])}", (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            if time.time() - last_hand_seen_time > RESET_DELAY:
                target = HOME_ANGLES[ELBOW_SERVO]
                smoothed = smooth_angle(current_angles[ELBOW_SERVO], target, alpha=0.12)
                filtered = apply_deadzone(current_angles[ELBOW_SERVO], smoothed, deadzone=1.0)
                current_angles[ELBOW_SERVO] = filtered

                send_angles_to_robot(ser, current_angles)

            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Robot Arm Hand Control - Bluetooth", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cleanup_and_exit()

cleanup_and_exit()
