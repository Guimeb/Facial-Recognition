import cv2
import os
import numpy as np
from datetime import datetime

# ---------- CONFIGURAÇÃO ----------
PERSON_NAME = "Guilherme"
DATA_PATH = "dataset"
MAX_IMAGES = 120            # quantas imagens você quer no total
STABLE_REQUIRED = 8         # quantos frames estáveis antes de salvar
MIN_FACE_WIDTH_RATIO = 0.18 # face deve ter >= 18% da largura do frame
CENTER_TOL = 25             # tolerância em pixels para considerar o rosto 'estável'
MIN_DIFF_TO_LAST = 25.0     # MAE mínimo em pixel para considerar imagem nova
AUGMENT = True              # cria 2 augmentações leves por captura
# -----------------------------------

person_path = os.path.join(DATA_PATH, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)

cap = cv2.VideoCapture(0)  # mantém do seu jeito
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = len(os.listdir(person_path))  # conta já existentes
stable_count = 0
last_center = None
last_w = None
last_saved_face = None
saved_files = []  # para poder desfazer

def save_face(img, idx, suffix=""):
    filename = f"{PERSON_NAME}_{idx:03d}{suffix}.jpg"
    path = os.path.join(person_path, filename)
    cv2.imwrite(path, img)
    return path

def augment_and_save(img, idx, base_idx):
    # pequenas rotações e ajuste de brilho
    saved = []
    rows, cols = img.shape
    for i, angle in enumerate((-12, 12)):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rot = cv2.warpAffine(img, M, (cols, rows))
        path = save_face(rot, base_idx + i + 1, suffix=f"_rot{angle}")
        saved.append(path)
    for i, alpha in enumerate((0.85, 1.15)):
        bright = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        path = save_face(bright, base_idx + 3 + i, suffix=f"_b{int(alpha*100)}")
        saved.append(path)
    return saved

print("Iniciando captura. Pressione 'q' para sair, 'c' para capturar manual, 'u' para desfazer última.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Erro: não foi possível ler frame. Tentando novamente...")
        continue

    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta faces (pega as maiores caso haja várias)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    msg = ""
    if len(faces) == 0:
        stable_count = 0
        last_center = None
        msg = "Nenhuma face detectada"
    else:
        # escolhe a maior face (pessoa mais próxima)
        faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        (x, y, w, h) = faces[0]
        center = (x + w//2, y + h//2)

        # checa tamanho (evita salvar rostos muito pequenos)
        if w < MIN_FACE_WIDTH_RATIO * w_frame:
            stable_count = 0
            msg = "Aproxime-se da camera"
        else:
            # estabilidade: compara centro/width com último
            if last_center is None:
                stable_count = 1
            else:
                dx = abs(center[0] - last_center[0])
                dy = abs(center[1] - last_center[1])
                dw = abs(w - last_w if last_w is not None else 0)
                if dx <= CENTER_TOL and dy <= CENTER_TOL and dw <= 15:
                    stable_count += 1
                else:
                    stable_count = 1

            last_center = center
            last_w = w
            msg = f"Face detectada - estabilidade {stable_count}/{STABLE_REQUIRED}"

            # se estável o suficiente, tenta salvar
            if stable_count >= STABLE_REQUIRED and count < MAX_IMAGES:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))

                # evita duplicatas comparando com última salva (MAE)
                is_new = True
                if last_saved_face is not None:
                    mae = np.mean(np.abs(roi.astype(np.int16) - last_saved_face.astype(np.int16)))
                    if mae < MIN_DIFF_TO_LAST:
                        is_new = False
                        msg = f"Igual à última (MAE={mae:.1f}) — pulando"

                if is_new:
                    # salva original
                    path = save_face(roi, count)
                    saved_files.append(path)
                    last_saved_face = roi.copy()
                    count += 1
                    msg = f"Salvou {os.path.basename(path)} ({count}/{MAX_IMAGES})"

                    # augment opcional (gera até 4 imagens extras, controla contador)
                    if AUGMENT and count < MAX_IMAGES:
                        aug_saved = augment_and_save(roi, count, count)
                        saved_files.extend(aug_saved)
                        count += len(aug_saved)
                        msg += f" +{len(aug_saved)} augment."

                stable_count = 0  # reset após salvar to evitar saves em sequência

        # desenha retângulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, center, 3, (0,255,0), -1)

    # overlay de informações
    cv2.putText(frame, f"Pessoa: {PERSON_NAME}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Salvas: {count}/{MAX_IMAGES}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, msg, (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

    cv2.imshow("Captura de rostos (melhorada)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # captura manual (salva mesmo que não esteja estável)
        if len(faces) > 0:
            x,y,w,h = faces[0]
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            path = save_face(roi, count)
            saved_files.append(path)
            last_saved_face = roi.copy()
            count += 1
            print(f"Captura manual: {path}")
    elif key == ord('u'):  # desfazer última
        if saved_files:
            last = saved_files.pop()
            try:
                os.remove(last)
                print("Removido:", last)
                count = max(0, count - 1)
                # tenta recarregar last_saved_face (simples: set None)
                last_saved_face = None
            except Exception as e:
                print("Erro ao remover:", e)

    # para segurança: se lote completo, para automaticamente
    if count >= MAX_IMAGES:
        print("Alvo de imagens alcançado.")
        break

cap.release()
cv2.destroyAllWindows()
