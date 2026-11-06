import time
import cv2
import mediapipe as mp
mp_hands = mp.tasks.vision.HandLandmarksConnections

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# holds the most recent GestureRecogniserResult.
latest_result = None

# Called by MediaPipe on a background thread whenever a frame finishes.
def on_result(result, output_image, timestamp_ms):
    """
    :param result: GestureRecognizerResult object for the frame. Holds gestures (ranked labels and scores per hand). handeddness and hand_landmarks
    :param output_image:mp.Image that was processed. UNUSED by function
    :param timestamp_ms: timestamp passed to recognise_async. Unused by the function.
    :return: Nothing.

    For each detected hand, it takes the top ranked gesture, skips None and anything below 50% confidence. #
    Looks up whether hand is Left or Right and prints result
    """
    global latest_result
    latest_result = result
    for i, gestures in enumerate(result.gestures):
        if not gestures:
            continue
        top = gestures[0]
        if top.category_name == "None" or top.score < 0.5:
            continue
        hand = result.handedness[i][0].category_name
        print(f"{hand} hand: {top.category_name} ({top.score:.2f})")
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="resources/gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

recognizer = GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)  # 0 is the default webcam

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # OpenCV gives BGR; MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must increase with each call
        timestamp_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, timestamp_ms)

        display = frame.copy()
        h, w, _ = display.shape

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Draw the connecting lines
                for conn in mp_hands.HAND_CONNECTIONS:
                    start = hand_landmarks[conn.start]
                    end = hand_landmarks[conn.end]
                    p1 = (int(start.x * w), int(start.y * h))
                    p2 = (int(end.x * w), int(end.y * h))
                    cv2.line(display, p1, p2, (0, 255, 0), 2)

                # Draw the 21 points
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imshow("Gesture check", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()