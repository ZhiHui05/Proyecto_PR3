import cv2
import numpy as np

# ==========================
# CONFIGURACIÓN
# ==========================

CAMERA_INDEX = 0  # 0 suele ser la webcam del portátil

# Área mínima y máxima del croissant en píxeles
AREA_MIN = 8000
AREA_MAX = 90000

# Umbrales de brillo/color.
# Se ajustan según vuestra iluminación.
BRILLO_MIN_OK = 80
BRILLO_MAX_OK = 170

# Relación de forma aproximada
# 1.0 = casi circular/cuadrado
# valores mayores = más alargado
ASPECT_RATIO_MIN = 1.1
ASPECT_RATIO_MAX = 4.5

# ==========================
# FUNCIONES
# ==========================

def clasificar_croissant(frame, contour):
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0

    # Máscara solo del croissant
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Color medio dentro del croissant
    mean_bgr = cv2.mean(frame, mask=mask)
    b, g, r = mean_bgr[:3]

    # Brillo aproximado
    brillo = (r + g + b) / 3

    # Clasificación
    if area < AREA_MIN:
        estado = "SIN PRODUCTO / MUY PEQUENO"
        color_estado = (0, 0, 255)
    elif area > AREA_MAX:
        estado = "DEMASIADO GRANDE / DOS PIEZAS"
        color_estado = (0, 0, 255)
    elif aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
        estado = "DEFORMADO"
        color_estado = (0, 165, 255)
    elif brillo < BRILLO_MIN_OK:
        estado = "MUY OSCURO / QUEMADO"
        color_estado = (0, 0, 255)
    elif brillo > BRILLO_MAX_OK:
        estado = "MUY CLARO / CRUDO"
        color_estado = (0, 165, 255)
    else:
        estado = "OK"
        color_estado = (0, 255, 0)

    datos = {
        "area": area,
        "brillo": brillo,
        "aspect_ratio": aspect_ratio,
        "bbox": (x, y, w, h),
        "estado": estado,
        "color_estado": color_estado
    }

    return datos


# ==========================
# PROGRAMA PRINCIPAL
# ==========================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: no se pudo abrir la webcam.")
    exit()

print("Sistema iniciado.")
print("Pulsa 'q' para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: no se pudo leer la imagen.")
        break

    # Redimensionar para trabajar más cómodo
    frame = cv2.resize(frame, (800, 600))

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ==========================
    # SEGMENTACIÓN DEL CROISSANT
    # ==========================
    # Rango aproximado para colores marrón/dorado
    # Puede necesitar ajuste según la luz
    lower_croissant = np.array([5, 40, 40])
    upper_croissant = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_croissant, upper_croissant)

    # Limpieza de ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Buscar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    estado_general = "SIN CROISSANT"
    color_general = (255, 255, 255)

    if contours:
        # Contorno más grande = posible croissant
        main_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(main_contour) > AREA_MIN:
            datos = clasificar_croissant(frame, main_contour)

            x, y, w, h = datos["bbox"]
            estado_general = datos["estado"]
            color_general = datos["color_estado"]

            # Dibujar rectángulo y contorno
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_general, 2)
            cv2.drawContours(frame, [main_contour], -1, color_general, 2)

            # Mostrar datos
            cv2.putText(frame, f"Estado: {datos['estado']}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_general, 2)

            cv2.putText(frame, f"Area: {int(datos['area'])}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Brillo: {int(datos['brillo'])}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Forma: {datos['aspect_ratio']:.2f}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "SIN CROISSANT", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "SIN CROISSANT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Mostrar ventanas
    cv2.imshow("Control de calidad - Croissants", frame)
    cv2.imshow("Mascara deteccion", mask)

    # Salir con q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()