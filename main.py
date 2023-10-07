import cv2
import numpy as np
import matplotlib.pyplot as plt

# Videó beolvasása
video_path = 'path_to_your_video.mp4'  # Cseréld le a videó elérési útvonalára
cap = cv2.VideoCapture(video_path)

# Ellenőrizd, hogy a videó megnyílt-e
if not cap.isOpened():
    print("Hiba: A videó nem nyitható meg.")
    exit()

# Első képkocka beolvasása és inicializálása
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Sebességadatok tárolásához listák inicializálása
velocities_x = []
velocities_y = []

# Az első 10 képkocka feldolgozása
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optikai áramlás kiszámítása
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None)

    # Sebességek kiszámítása
    velocities = curr_points - prev_points
    vx, vy = np.mean(velocities, axis=0)

    # Sebességek hozzáadása a listákhoz
    velocities_x.append(vx)
    velocities_y.append(vy)

    # Az aktuális képkockát az előzőhöz állítjuk be a következő ciklusban
    prev_gray = frame_gray.copy()
    prev_points = curr_points

# Sebességplot készítése
plt.figure(figsize=(10, 5))
plt.plot(range(10), velocities_x, label='X irányú sebesség')
plt.plot(range(10), velocities_y, label='Y irányú sebesség')
plt.xlabel('Képkocka sorszáma')
plt.ylabel('Sebesség (pixelek/s)')
plt.legend()
plt.title('Sebességplot az első 10 képkockára')
plt.grid(True)

# Plot megjelenítése
plt.show()

# Videókapcsolat bezárása
cap.release()
cv2.destroyAllWindows()
