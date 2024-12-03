import cv2
import numpy as np

# Bild laden
image = cv2.imread('Testimage.jpg')

# Bild von BGR zu HSV konvertieren
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Grenzen für die blaue Farbe definieren
lower_blue = np.array([100, 43, 127])
upper_blue = np.array([124, 255, 255])

# Maske erstellen, die nur die blauen Bereiche enthält
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Anwenden der Maske auf das Originalbild
result = cv2.bitwise_and(image, image, mask=mask)

# Finden des Moments der Maske, um den Mittelpunkt zu berechnen
moments = cv2.moments(mask)
if moments['m00'] != 0:
    cX = int(moments['m10'] / moments['m00'])
    cY = int(moments['m01'] / moments['m00'])
    # Zeichnen des Mittelpunkts auf das Ergebnisbild
    cv2.circle(result, (cX, cY), 5, (0, 255, 0), -1)
    print(f"Mittelpunkt des blauen Gegenstands: ({cX}, {cY})")

    # Berechnung des umgebenden Rechtecks für die Maske
    x, y, w, h = cv2.boundingRect(mask)
    # Zeichnen des Rechtecks auf das Ergebnisbild
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Beschriften des Rechtecks mit Farbe und Form des Objekts
    label = "Blaues Objekt (Rund)"
    cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
else:
    print("Kein blauer Gegenstand gefunden.")

# Ergebnis anzeigen
cv2.imshow('Original', image)
cv2.imshow('Blauer Gegenstand', result)

# Warten, bis eine Taste gedrückt wird, und Fenster schließen
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Das Ergebnis speichern
#cv2.imwrite('filtered_blue_object.jpg', result)
