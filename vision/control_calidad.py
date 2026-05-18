import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# ============================================================
# ROBOFLOW
# ============================================================

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZPQZ80VwNXxldk75JfXj"
)

MODEL_ID = "bread-dsgs5-4w3ls/1"

# ============================================================
# CAMARA
# ============================================================

cap = cv2.VideoCapture(2)

# ============================================================
# DETECTOR DE QUEMADO
# ============================================================

def is_burned(crop):

    # convertir a HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # rango de negro
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 110])

    # máscara negra
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # porcentaje negro
    black_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    black_ratio = black_pixels / total_pixels

    print("BLACK RATIO:", round(black_ratio, 2))

    # threshold
    return black_ratio > 0.25

# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # GUARDAR FRAME TEMPORAL
    # --------------------------------------------------------

    cv2.imwrite("temp.jpg", frame)

    # --------------------------------------------------------
    # INFERENCIA ROBOFLOW
    # --------------------------------------------------------

    result = CLIENT.infer(
        "temp.jpg",
        model_id=MODEL_ID
    )

    predictions = result["predictions"]

    # --------------------------------------------------------
    # DIBUJAR RESULTADOS
    # --------------------------------------------------------

    for pred in predictions:

        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        conf = pred["confidence"]

        # coordenadas bbox
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # evitar coordenadas negativas
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # crop del cruasán
        crop = frame[y1:y2, x1:x2]

        burned = False

        if crop.size > 0:
            burned = is_burned(crop)

        # ----------------------------------------------------
        # RESULTADO FINAL
        # ----------------------------------------------------

        if burned:
            color = (0, 0, 255)
            label = f"CRUASAN QUEMADO {conf:.2f}"
        else:
            color = (0, 255, 0)
            label = f"CRUASAN OK {conf:.2f}"

        # bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # texto
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # --------------------------------------------------------
    # MOSTRAR
    # --------------------------------------------------------

    cv2.imshow("Detector de Cruasanes", frame)

    # salir con Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# LIMPIEZA
# ============================================================

cap.release()
cv2.destroyAllWindows()
