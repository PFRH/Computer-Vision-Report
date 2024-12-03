import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_isolate_face(image_path):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("Error: Haar Cascade XML file not found.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
        return

    # Create a mask that blacks out everything except the face
    mask = np.zeros_like(image)

    for (x, y, w, h) in faces:
        # Draw the rectangle for the detected face on the mask
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]

    # Blacken the image outside of the face
    isolated_face_image = mask

    # Plot the original and modified versions of the image
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Isolated face
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(isolated_face_image, cv2.COLOR_BGR2RGB))
    plt.title('Isolated Face')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path to the image
    image_path = "Face_01.jpg"
    detect_and_isolate_face(image_path)
