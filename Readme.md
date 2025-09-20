# Reconhecimento Facial - Projeto IOT & IOB

## Descrição do Projeto

Este projeto realiza **reconhecimento e identificação facial** usando OpenCV e o algoritmo **LBPH** (Local Binary Patterns Histograms). O objetivo é capturar imagens de um usuário, treinar um modelo local e reconhecer rostos em tempo real, exibindo retângulos e nomes sobre as faces detectadas.

## Dependências

Certifique-se de ter instalado:

```
pip install opencv-contrib-python numpy
```

> Obs: É importante instalar o **opencv-contrib-python**, pois ele contém `cv2.face` necessário para o LBPH.

---

## Ordem de Execução

1. **Captura de imagens (`captura.py`)**

   * Executa a câmera para capturar rostos do usuário.
   * O programa salva automaticamente as imagens quando o rosto está estável.
   * Teclas de atalho:

     * `q`: sair da captura
     * `c`: captura manual (salva mesmo que não esteja estável)
     * `u`: desfazer última imagem salva

   ```
   python captura.py
   ```

   * As imagens serão salvas em `dataset/<NOME_DO_USUARIO>/`.

2. **Treinamento do modelo (`treino.py`)**

   * Treina o LBPH com as imagens capturadas.
   * Gera dois arquivos:

     * `face_model.yml` → modelo treinado
     * `labels.npy` → labels do dataset

   ```
   python treino.py
   ```

3. **Reconhecimento facial em tempo real (`facial_recognition.py`)**

   * Executa a câmera e realiza o reconhecimento em tempo real.
   * Exibe retângulos sobre as faces detectadas e o nome identificado.

   ```
   python facial_recognition.py
   ```

---

## Parâmetros Relevantes

* **THRESHOLD**: Limite de confiança do LBPH (quanto menor, mais rigoroso).
* **SCALE\_FACTOR / MIN\_NEIGHBORS**: Ajustam a detecção Haar Cascade.
* **MIN\_FACE\_SIZE\_RATIO**: Define tamanho mínimo do rosto para ser detectado.
* **STABLE\_FRAMES**: Número de frames consecutivos para confirmar uma identificação.

> Ajustando esses parâmetros é possível observar como a detecção e identificação mudam, tornando o modelo mais ou menos sensível.

---

## Considerações Éticas

* Este projeto utiliza imagens faciais.
* **Não utilize rostos de terceiros sem consentimento.**
* Evite armazenar dados pessoais em sistemas públicos ou inseguros.