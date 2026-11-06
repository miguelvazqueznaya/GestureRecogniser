import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="resources/gesture_recognizer.task"),
    running_mode=VisionRunningMode.IMAGE,
    #running_mode=VisionRunningMode.LIVE_STREAM,
    #running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

##IMAGE DETECTION
with GestureRecognizer.create_from_options(options) as recognizer:
    # Synchronous single-image call (stateless across images)
    mp_image = mp.Image.create_from_file("resources/thumbs_up2.jpg")
    result = recognizer.recognize(mp_image)

    if not result.gestures:
        print("No hands detected.")
    for i, gestures in enumerate(result.gestures):
        top = gestures[0]
        hand = result.handedness[i][0].category_name
        print(f"{hand} hand: {top.category_name} ({top.score:.2f})")
