import cv2
import os
import numpy as np

DATA_PATH = "dataset"

people = [p for p in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, p))]

faces_data = []
labels = []
label_dict = {}
label = 0

IMG_SIZE = (200, 200)  # mesmo tamanho usado na captura

for person in people:
    person_path = os.path.join(DATA_PATH, person)
    files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not files:
        print(f"Aviso: nenhuma imagem encontrada para {person}, ignorando.")
        continue

    label_dict[label] = person
    print(f"Carregando {len(files)} imagens de {person} (label {label})")

    for file in files:
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Aviso: não foi possível ler {img_path}, ignorando.")
            continue

        # garante tamanho consistente
        if img.shape != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)

        faces_data.append(img)
        labels.append(label)

    label += 1

faces_data = np.array(faces_data)
labels = np.array(labels)

print(f"\nTotal de imagens: {len(faces_data)}")
print(f"Classes: {label_dict}")

# Criar e treinar o reconhecedor LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces_data, labels)

# Salvar modelo e labels
face_recognizer.save("face_model.yml")
np.save("labels.npy", label_dict)

print("\nTreinamento concluído e modelo salvo como 'face_model.yml'")
