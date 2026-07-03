import cv2

VIDEO_PATH = "/Users/zhaojiacheng/Desktop/badminton_3d_construction/data/video/test1.mp4"  # 改成你的视频路径

points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: [{x}, {y}]")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("无法读取视频首帧，请检查路径")

show = frame.copy()
cv2.namedWindow("Click 4 court corners", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click 4 court corners", 1280, 720)
cv2.setMouseCallback("Click 4 court corners", on_mouse)

while True:
    disp = show.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(disp, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(
        disp,
        "Click order: LeftTop -> RightTop -> LeftBottom -> RightBottom | R:reset Q:quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Click 4 court corners", disp)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        points = []
        print("Reset points.")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

if len(points) == 4:
    print("\nCOURT_CORNERS = [")
    for p in points:
        print(f"    [{p[0]}, {p[1]}],")
    print("]")
else:
    print(f"\n你只点了 {len(points)} 个点，需要4个。")