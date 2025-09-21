import cv2
import numpy as np
import os
import sys

# ---------- CONFIGURAÇÃO ----------
MODEL_PATH = "face_model.yml"     # caminho do modelo treinado (LBPH)
LABELS_PATH = "labels.npy"        # labels salvas (dicionário id -> nome)

THRESHOLD = 55                    # limite de confiança (quanto menor, mais restritivo)
SCALE_FACTOR = 1.1                # fator de escala para Haar Cascade
MIN_NEIGHBORS = 5                 # vizinhos mínimos para validação da detecção
MIN_FACE_SIZE_RATIO = 0.15        # tamanho mínimo relativo do rosto (15% da tela)
STABLE_FRAMES = 5                 # nº de frames iguais para confirmar identificação
# ---------------------------------

if not hasattr(cv2, "face"):
    print("ERRO: cv2.face não encontrado. Instale: pip install opencv-contrib-python")
    sys.exit(1)

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("Modelo ou labels não encontrados.")
    sys.exit(1)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

label_dict = np.load(LABELS_PATH, allow_pickle=True).item()
label_dict = {int(k): v for k, v in label_dict.items()}
print("Labels carregadas:", label_dict)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Não foi possível abrir a câmera.")

face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

prev_label = None
stable_count = 0

print("Iniciando reconhecimento. Pressione 'q' para sair.")

while True:

    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_size = (int(w * MIN_FACE_SIZE_RATIO), int(h * MIN_FACE_SIZE_RATIO))

    # Detecta rostos frontais e de perfil
    faces_front = face_cascade_front.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS, minSize=min_size)
    faces_profile = face_cascade_profile.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS, minSize=min_size)
    faces = list(faces_front) + list(faces_profile)

    # Processa cada rosto encontrado e ignora rostos menores
    for (x, y, w, h) in faces:

        if w < min_size[0] or h < min_size[1]:
            continue

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = face_recognizer.predict(face)

        if confidence < THRESHOLD:
            name = label_dict.get(label, "Desconhecido")
            color = (0, 255, 0)

            if prev_label == label:
                stable_count += 1
            else:
                stable_count = 1
            prev_label = label

            if stable_count < STABLE_FRAMES:
                name = "Reconhecendo..."
        else:
            name = "Desconhecido"
            color = (0, 0, 255)
            prev_label = None
            stable_count = 0

        cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Reconhecimento Facial - LBPH", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #sair
        break

cap.release()
cv2.destroyAllWindows()
