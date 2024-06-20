import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Función para visualizar las detecciones
def visualize(image, detection_result, person_limit):
    person_count = 0
    for detection in detection_result.detections:
        category = detection.categories[0]
        
        # Filtrar solo las detecciones de personas
        if category.category_name == 'person':
            person_count += 1
            bbox = detection.bounding_box
            caption = f'{category.category_name} ({int(category.score*100)}%)'
            image = cv2.rectangle(image, (bbox.origin_x, bbox.origin_y), 
                                  (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), 
                                  (0, 255, 0), 2)
            image = cv2.putText(image, caption, (bbox.origin_x, bbox.origin_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Cambiar el color de la pantalla según el conteo de personas
    if person_count >= person_limit:
        overlay_color = (0, 0, 255)  # Rojo
    else:
        overlay_color = (0, 255, 0)  # Verde
    
    # Crear una imagen con el color de superposición
    overlay = np.full(image.shape, overlay_color, dtype=np.uint8)
    alpha = 0.3  # Transparencia del color de superposición
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Mostrar el conteo de personas en la imagen
    cv2.putText(image, f'Persons: {person_count} of {PERSON_LIMIT}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

# Función para mostrar la flecha o la línea en la tercera pantalla
def display_direction(person_count1, person_count2, person_limit):
    # Crear una imagen en negro
    direction_image = np.zeros((160, 213, 3), dtype=np.uint8)
    
    if person_count1 >= person_limit and person_count2 >= person_limit:
        # Ambos alcanzan el límite, mostrar una línea roja
        cv2.line(direction_image, (0, 80), (213, 80), (0, 0, 255), 5)
    elif person_count1 < person_limit and person_count2 < person_limit:
        # Ninguno alcanza el límite, mostrar una flecha de dos puntas
        cv2.arrowedLine(direction_image, (106, 80), (53, 80), (0, 255, 0), 5)
        cv2.arrowedLine(direction_image, (106, 80), (159, 80), (0, 255, 0), 5)
    elif person_count1 < person_limit:
        # Cámara 1 no alcanza el límite, mostrar flecha a la izquierda
        cv2.arrowedLine(direction_image, (106, 80), (53, 80), (0, 255, 0), 5)
    else:
        # Cámara 2 no alcanza el límite, mostrar flecha a la derecha
        cv2.arrowedLine(direction_image, (106, 80), (159, 80), (0, 255, 0), 5)
    
    return direction_image

# Límite de personas
PERSON_LIMIT = 2

# STEP 1: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 2: Open the cameras.
cap1 = cv2.VideoCapture(0)  # Primera cámara
cap2 = cv2.VideoCapture(1)  # Segunda cámara

while True:
    # STEP 3: Read a frame from the first camera.
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    # STEP 4: Read a frame from the second camera.
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    # STEP 5: Convert the frames to MediaPipe Image format.
    image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1)
    image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2)

    # STEP 6: Detect objects in the frames.
    detection_result1 = detector.detect(image1)
    detection_result2 = detector.detect(image2)

    # STEP 7: Visualize the detection results with the person limit.
    annotated_frame1 = visualize(frame1, detection_result1, PERSON_LIMIT)
    annotated_frame2 = visualize(frame2, detection_result2, PERSON_LIMIT)

    # STEP 8: Count persons in each frame.
    person_count1 = sum(1 for detection in detection_result1.detections if detection.categories[0].category_name == 'person')
    person_count2 = sum(1 for detection in detection_result2.detections if detection.categories[0].category_name == 'person')

    # STEP 9: Display direction based on person counts.
    direction_image = display_direction(person_count1, person_count2, PERSON_LIMIT)

    # STEP 10: Display the annotated frames and direction image.
    cv2.imshow('Object Detection - Camera 1', annotated_frame1)
    cv2.imshow('Object Detection - Camera 2', annotated_frame2)
    cv2.imshow('Direction', direction_image)

    # STEP 11: Exit if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 12: Release resources.
cap1.release()
cap2.release()
cv2.destroyAllWindows()
