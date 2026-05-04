import cv2
import numpy as np

# ==========================
# CONFIGURACION
# ==========================

CAMERA_INDEX = 0  # 0 suele ser la webcam del portatil

# Area minima y maxima del croissant en pixeles.
AREA_MIN = 8000
AREA_MAX = 90000

# Limites para descartar fondos u objetos enormes antes de clasificar.
AREA_CANDIDATO_MAX = int(AREA_MAX * 1.25)
BBOX_FRAME_MAX = 0.70
FILL_RATIO_MIN = 0.25
FILL_RATIO_MAX = 0.85

# Umbrales de brillo/color.
# Se ajustan segun vuestra iluminacion.
BRILLO_MIN_OK = 80
BRILLO_MAX_OK = 170

# Relacion de forma aproximada:
# 1.0 = casi circular/cuadrado; valores mayores = mas alargado.
ASPECT_RATIO_MIN = 1.1
ASPECT_RATIO_MAX = 4.5

# HSV del pan tostado. Subir VALOR_MIN ayuda a ignorar negro/sombras.
H_MIN = 5
H_MAX = 35
SAT_MIN = 55
VALOR_MIN = 70

# ==========================
# FUNCIONES
# ==========================


def datos_contorno(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    fill_ratio = area / (w * h) if w * h != 0 else 0

    return {
        "area": area,
        "bbox": (x, y, w, h),
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio
    }


def crear_mascara_croissant(hsv):
    lower_croissant = np.array([H_MIN, SAT_MIN, VALOR_MIN])
    upper_croissant = np.array([H_MAX, 255, 255])

    mask = cv2.inRange(hsv, lower_croissant, upper_croissant)

    # El tono en pixeles negros es inestable; se elimina por brillo.
    _, _, value = cv2.split(hsv)
    mask_no_negro = cv2.inRange(value, VALOR_MIN, 255)
    mask = cv2.bitwise_and(mask, mask_no_negro)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def es_candidato_croissant(contour, frame_shape):
    datos = datos_contorno(contour)
    area = datos["area"]
    x, y, w, h = datos["bbox"]
    aspect_ratio = datos["aspect_ratio"]
    fill_ratio = datos["fill_ratio"]
    frame_h, frame_w = frame_shape[:2]

    if area < AREA_MIN or area > AREA_CANDIDATO_MAX:
        return False
    if w > frame_w * BBOX_FRAME_MAX or h > frame_h * BBOX_FRAME_MAX:
        return False
    if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
        return False
    if fill_ratio < FILL_RATIO_MIN or fill_ratio > FILL_RATIO_MAX:
        return False

    return True


def elegir_mejor_candidato(contours, frame_shape):
    candidatos = [
        contour for contour in contours
        if es_candidato_croissant(contour, frame_shape)
    ]

    if not candidatos:
        return None

    area_objetivo = (AREA_MIN + AREA_MAX) / 2
    return min(candidatos, key=lambda contour: abs(cv2.contourArea(contour) - area_objetivo))


def clasificar_croissant(frame, contour):
    datos_base = datos_contorno(contour)
    area = datos_base["area"]
    x, y, w, h = datos_base["bbox"]
    aspect_ratio = datos_base["aspect_ratio"]

    # Mascara solo del croissant.
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Color medio dentro del croissant.
    mean_bgr = cv2.mean(frame, mask=mask)
    b, g, r = mean_bgr[:3]

    # Brillo aproximado.
    brillo = (r + g + b) / 3

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

    # Redimensionar para trabajar mas comodo.
    frame = cv2.resize(frame, (800, 600))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ==========================
    # SEGMENTACION DEL CROISSANT
    # ==========================
    mask = crear_mascara_croissant(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    main_contour = elegir_mejor_candidato(contours, frame.shape)

    if main_contour is not None:
        datos = clasificar_croissant(frame, main_contour)

        x, y, w, h = datos["bbox"]
        color_general = datos["color_estado"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color_general, 2)
        cv2.drawContours(frame, [main_contour], -1, color_general, 2)

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

    cv2.imshow("Control de calidad - Croissants", frame)
    cv2.imshow("Mascara deteccion", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
