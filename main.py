import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 獲取多目標追蹤器實例，具體要使用什麼算法 add 指定
trackers = cv2.MultiTracker_create()

while True:
    _, frame = cap.read()
    if frame is None:
        break

    # 每次更新幀都會獲取要追蹤的目標
    (success, boxes) = trackers.update(frame)
    for box in boxes:
        (x1, y1, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0))

        # 畫出視窗中心的方框
        height, width = frame.shape[:2]
        start_row, start_col = int(height / 2 - 50), int(width / 2 - 50) # 方框的大小為 100x100
        end_row, end_col = int(height / 2 + 50), int(width / 2 + 50)
        cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)
        
        # 追蹤區域的中心與視窗中心繪製一條線
        p1 = (x1 + w // 2, y1 + h // 2)
        p2 = (width // 2, height // 2)
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # 計算線段長度
        line_length = np.linalg.norm(np.array(p1)-np.array(p2))
        cv2.putText(frame, 'Line length: {:.2f}'.format(line_length), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 在視窗中顯示 x 和 y 軸的差值
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        if abs(dx) < 100:
            cv2.putText(frame, 'dx: {}'.format(dx), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'dx: {}'.format(dx), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if abs(dy) < 100:
            cv2.putText(frame, 'dy: {}'.format(dy), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'dy: {}'.format(dy), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # box 是個 Rect2d，運行至此你需要用滑鼠選取一塊區域
        box = cv2.selectROI("video", frame, True, False)
        # 去追蹤 frame 上的某個區域物體
        trackers.add(cv2.TrackerKCF_create(), frame, box)

    if key == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()
