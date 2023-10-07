import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Képsorozat elérési útvonala (a képeknek ugyanazt a könyvtárat kell tartalmazniuk)
image_sequence_path = 'images'  # Cseréld le a képsorozat könyvtárának elérési útvonalára

# Ellenőrizzük, hogy az elérési útvonal létezik-e
if not os.path.exists(image_sequence_path):
    print("Hiba: A megadott elérési útvonal nem létezik.")
    exit()

# Képek beolvasása a sorozatból és sebességkiszámítás
velocities_x = []
velocities_y = []

# Képsorozat feldolgozása
image_files = sorted(os.listdir(image_sequence_path))
prev_image = None

for image_file in image_files:
    # Kép beolvasása
    image_path = os.path.join(image_sequence_path, image_file)
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    # Szürkeárnyalatos konverzió
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Az optikai áramlás kiszámítása a képeken
    if prev_image is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_image, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vx = np.mean(flow[..., 0])
        vy = np.mean(flow[..., 1])
        velocities_x.append(vx)
        velocities_y.append(vy)

    prev_image = frame_gray

# Sebességplot készítése
plt.figure(figsize=(10, 5))
plt.plot(range(len(velocities_x)), velocities_x, label='X irányú sebesség')
plt.plot(range(len(velocities_y)), velocities_y, label='Y irányú sebesség')
plt.xlabel('Képkocka sorszáma')
plt.ylabel('Sebesség (pixelek/s)')
plt.legend()
plt.title('Sebességplot a képsorozatra')
plt.grid(True)

# Plot megjelenítése
plt.show()
