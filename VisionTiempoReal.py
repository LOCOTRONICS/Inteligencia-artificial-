import cv2
from ultralytics import YOLO

# 1. Cargar el modelo
model = YOLO('pega aqui la ruta de tu .pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True, conf=0.4)

    alarma_activa = False

    for r in results:
        # Dibujar las detecciones
        annotated_frame = r.plot()
        
    cv2.imshow("Deteccion", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()