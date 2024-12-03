import cv2
import numpy as np

# Bild laden
image = cv2.imread('Testimage.jpg')

# Bild von BGR zu HSV konvertieren
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Grenzen für die blaue Farbe definieren
lower_blue = np.array([100, 43, 127])
upper_blue = np.array([124, 255, 255])
# Grenzen für die rote Farbe definieren
lower_red1 = np.array([0, 43, 127])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 43, 127])
upper_red2 = np.array([180, 255, 255])
# Grenzen für die grüne Farbe definieren
lower_green = np.array([35, 43, 127])
upper_green = np.array([85, 255, 255])

# Maske für blaue Bereiche erstellen
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
# Maske für rote Bereiche erstellen (da Rot in zwei Bereichen im HSV ist)
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
# Maske für grüne Bereiche erstellen
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Gesamtergebnisbild erstellen und Hintergrund schwarz machen
result = np.zeros_like(image)

# Funktion zum Finden von Mittelpunkt und Zeichnen des Rechtecks sowie Beschriftung
def process_mask(mask, color_label, shape_label, color_rgb):
    # Anwenden der Maske auf das Originalbild, um die entsprechenden Bereiche freizulegen
    masked_object = cv2.bitwise_and(image, image, mask=mask)
    # Übertragen der farbigen Bereiche auf das Ergebnisbild
    result[mask > 0] = masked_object[mask > 0]
    
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        cX = int(moments['m10'] / moments['m00'])
        cY = int(moments['m01'] / moments['m00'])
        # Zeichnen des Mittelpunkts auf das Ergebnisbild
        cv2.circle(result, (cX, cY), 5, color_rgb, -1)
        print(f"Mittelpunkt des {color_label} Gegenstands: ({cX}, {cY})")

        # Berechnung des umgebenden Rechtecks für die Maske
        x, y, w, h = cv2.boundingRect(mask)
        # Zeichnen des Rechtecks auf das Ergebnisbild
        cv2.rectangle(result, (x, y), (x + w, y + h), color_rgb, 2)
        # Beschriften des Rechtecks mit Farbe und Form des Objekts
        label = f"{color_label} Objekt ({shape_label})"
        cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
    else:
        print(f"Kein {color_label} Gegenstand gefunden.")

# Verarbeiten der blauen Maske
process_mask(mask_blue, "Blaues", "Rund", (255, 0, 0))
# Verarbeiten der roten Maske
process_mask(mask_red, "Rotes", "Quadrat", (0, 0, 255))
# Verarbeiten der grünen Maske
process_mask(mask_green, "Gruenes", "Quadrat", (0, 255, 0))

# Ergebnis anzeigen
cv2.imshow('Original', image)
cv2.imshow('Identifizierte Gegenstände', result)

# Warten, bis eine Taste gedrückt wird, und Fenster schließen
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Das Ergebnis speichern
#cv2.imwrite('filtered_blue_object_all.jpg', result)
