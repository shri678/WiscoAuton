import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "input_image.jpg"
img = cv2.imread(image_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Range of values for red in the HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])


mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2


kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cone_centers = []


for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cone_centers.append((cx, cy))


cone_centers = sorted(cone_centers, key=lambda p: p[1])
left_cones, right_cones = [], []

mid_x = img.shape[1] // 2  #Middle of img

for (x, y) in cone_centers:
    if x < mid_x:
        left_cones.append((x, y))
    else:
        right_cones.append((x, y))

# Fit lines
def fit_line_ransac(points):
    if len(points) > 1:
        points = np.array(points)
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        return (vx, vy, x0, y0)
    return None

left_line = fit_line_ransac(left_cones)
right_line = fit_line_ransac(right_cones)

output_image = img.copy()

def draw_line(image, line, color=(0, 0, 255)):
    if line:
        vx, vy, x0, y0 = line
        y1, y2 = 0, image.shape[0]  # Extend to top and bottom of image
        x1 = int(x0 + (y1 - y0) * (vx / vy))
        x2 = int(x0 + (y2 - y0) * (vx / vy))
        cv2.line(image, (x1, y1), (x2, y2), color, 2)


draw_line(output_image, left_line)
draw_line(output_image, right_line)

#Write output of the code to `answer.png`
output_path = "answer.png"
cv2.imwrite(output_path, output_image)

#Display image at the end
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
