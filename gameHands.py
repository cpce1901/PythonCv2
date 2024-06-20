import cv2
import mediapipe as mp
import time
import random
import pygame

# Variables globales
game_duration = 120
fill_duration = 0.5  # Duración en segundos para llenar el cuadro con su propio color
points = 0
hand_inside_box = False
box_visible = False
extra_box_visible = False  # Variable para controlar la aparición del cuadro adicional
box_x, box_y = 0, 0
box_color = (255, 0, 0)  # Color inicial del cuadro
extra_box_color = (0, 0, 0)  # Color del cuadro adicional
last_toggle_time = time.time()
start_time = time.time()
current_color_index = 0
high_contrast_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
num_colors = len(high_contrast_colors)
levels = [
    {"score_range": (0, 2), "box_interval_time": 2500, "reaction_time": 1200},
    {"score_range": (3, 6), "box_interval_time": 2000, "reaction_time": 1000},
    {"score_range": (7, 9), "box_interval_time": 1500, "reaction_time": 800},
    {"score_range": (10, 12), "box_interval_time": 1000, "reaction_time": 600},
    {"score_range": (13, float('inf')), "box_interval_time": 700, "reaction_time": 400}
]

# Inicializar pygame para reproducir audio
pygame.mixer.init()
pygame.mixer.music.load('assets/music.mp3')
pygame.mixer.music.play()
positive_sound = pygame.mixer.Sound('assets/tada.mp3')
negative_sound = pygame.mixer.Sound('assets/boo.mp3')
start_game = pygame.mixer.Sound('assets/start.mp3')
end_game = pygame.mixer.Sound('assets/boom.mp3')

# Inicializar MediaPipe Hands y dibujar herramientas
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Definir dimensiones del cuadro
box_w, box_h = 80, 80


# Funciones
def play_sound(sound):
    sound.play()

def update_box(level, image, hand_landmarks):
    global box_x, box_y, box_color, box_visible
    
    frame_w = image.shape[1]  # Obtener el ancho del frame de la imagen
    frame_h = image.shape[0]  # Obtener la altura del frame de la imagen
    
    hand_positions = []
    if hand_landmarks:
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * frame_w), int(landmark.y * frame_h)
            hand_positions.append((cx, cy))
    
    while True:
        box_x = random.randint(0, frame_w - box_w)
        box_y = random.randint(0, frame_h - box_h)
        
        # Verificar que el cuadro esté al menos a 20 píxeles de todas las posiciones de las manos
        safe_distance = 50
        too_close = False
        for hand_pos in hand_positions:
            hand_x, hand_y = hand_pos
            if abs(box_x - hand_x) < safe_distance and abs(box_y - hand_y) < safe_distance:
                too_close = True
                break
        
        if not too_close:
            break
    
    box_color = random.choice(high_contrast_colors)
    box_visible = True
    return level["box_interval_time"], level["reaction_time"]

def hide_box():
    global box_visible
    box_visible = False

def maximize_window():
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def game_loop():
    global start_time, last_toggle_time, points, hand_inside_box, box_visible, extra_box_visible

    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        maximize_window()
        
        start_time = time.time()
        points = 0
        last_toggle_time = start_time
        box_interval_time, reaction_time = 0, 0

        while cap.isOpened():
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > game_duration:
                print(f"Tiempo terminado. Puntuación final: {points}")
                show_final_screen()
                break
            
            success, image = cap.read()
            if not success:
                print("Ignorando el frame vacío de la cámara.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_level = None
            for level in levels:
                if level["score_range"][0] <= points <= level["score_range"][1]:
                    current_level = level
                    break

            if current_level is None:
                print(f"No se encontró nivel para la puntuación {points}. Usando el nivel predeterminado.")
                current_level = levels[0]

            if current_time - last_toggle_time > current_level["box_interval_time"] / 1000:
                last_toggle_time = current_time
                hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
                box_interval_time, reaction_time = update_box(current_level, image, hand_landmarks)
                last_toggle_time = current_time

            if box_visible:
                if current_time - last_toggle_time > reaction_time / 1000:
                    if not hand_inside_box:
                        points -= 1
                        if points < 0:
                            points = 0
                        print(f"Puntos: {points} (Se ha excedido el tiempo de reacción)")
                        play_sound(negative_sound)
                        hide_box()
                        last_toggle_time = current_time
                else:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            for landmark in hand_landmarks.landmark:
                                h, w, _ = image.shape
                                cx, cy = int(landmark.x * w), int(landmark.y * h)
                                
                                if box_x < cx < box_x + box_w and box_y < cy < box_y + box_h:
                                    if not hand_inside_box:
                                        points += 1
                                        print(f"Puntos: {points}")
                                        play_sound(positive_sound)
                                        hand_inside_box = True
                                        # Mostrar el cuadro extra y configurar su color
                                        extra_box_visible = True
                                        extra_box_color = box_color
                                        extra_box_start_time = current_time
                                        hide_box()  # Ocultar el cuadro principal
                                        last_toggle_time = current_time + box_interval_time / 1000
                                    break
                            else:
                                hand_inside_box = False
                    else:
                        hand_inside_box = False
            else:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if box_visible:
                cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, 2)

            # Dibujar el cuadro extra si es visible
            if extra_box_visible:
                if current_time - extra_box_start_time < fill_duration:
                    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), extra_box_color, -1)
                else:
                    extra_box_visible = False

            cv2.putText(image, f'Puntos: {points}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            remaining_time_text = f'Tiempo restante: {int(game_duration - elapsed_time)}s'
            text_size, _ = cv2.getTextSize(remaining_time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = screen_width - text_size[0] - 10
            text_y = screen_height - 10
            cv2.putText(image, remaining_time_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Hand Tracking', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                start_game.play()
                break

def show_final_screen():
    global points
    button_width, button_height = 200, 80
    button_margin = 20
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 4
    button_replay = [(screen_center_x - button_width - button_margin, screen_center_y * 3),
                     (screen_center_x - button_margin, screen_center_y * 3 + button_height)]
    button_exit = [(screen_center_x + button_margin, screen_center_y * 3),
                   (screen_center_x + button_width + button_margin, screen_center_y * 3 + button_height)]

    replay_pressed = False
    exit_pressed = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        while True:
            success, image = cap.read()
            if not success:
                print("Ignorando el frame vacío de la cámara.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            text = f'Puntuacion Final: {points}'
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            text_x = (screen_width - text_size[0]) // 2
            cv2.putText(image, text, (text_x, screen_center_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Dibujar botones
            replay_color = (0, 255, 0) if not replay_pressed else (0, 200, 0)
            cv2.rectangle(image, button_replay[0], button_replay[1], replay_color, -1)
            text_replay = 'Reiniciar'
            text_size_replay, _ = cv2.getTextSize(text_replay, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x_replay = button_replay[0][0] + (button_width - text_size_replay[0]) // 2
            text_y_replay = button_replay[0][1] + (button_height + text_size_replay[1]) // 2
            text_color_replay = (255, 255, 255) if not replay_pressed else (150, 150, 150)
            cv2.putText(image, text_replay, (text_x_replay, text_y_replay), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color_replay, 2, cv2.LINE_AA)

            exit_color = (0, 0, 255) if not exit_pressed else (0, 0, 200)
            cv2.rectangle(image, button_exit[0], button_exit[1], exit_color, -1)
            text_exit = 'Salir'
            text_size_exit, _ = cv2.getTextSize(text_exit, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x_exit = button_exit[0][0] + (button_width - text_size_exit[0]) // 2
            text_y_exit = button_exit[0][1] + (button_height + text_size_exit[1]) // 2
            text_color_exit = (255, 255, 255) if not exit_pressed else (150, 150, 150)
            cv2.putText(image, text_exit, (text_x_exit, text_y_exit), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color_exit, 2, cv2.LINE_AA)

            replay_hand_present = False
            exit_hand_present = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = image.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    # Verificar si el dedo índice está sobre los botones
                    if button_replay[0][0] < cx < button_replay[1][0] and button_replay[0][1] < cy < button_replay[1][1]:
                        replay_hand_present = True
                        if not replay_pressed:
                            print("Botón Reiniciar presionado por primera vez")
                            replay_pressed = True
                    else:
                        if replay_pressed:
                            print("Botón Reiniciar soltado")
                            replay_pressed = False
                            start_game.play()
                            return True
                            # Realizar acciones de reiniciar juego aquí

                    if button_exit[0][0] < cx < button_exit[1][0] and button_exit[0][1] < cy < button_exit[1][1]:
                        exit_hand_present = True
                        if not exit_pressed:
                            print("Botón Salir presionado por primera vez")
                            exit_pressed = True
                    else:
                        if exit_pressed:
                            print("Botón Salir soltado")
                            exit_pressed = False
                            end_game.play()
                            return False
                            # Realizar acciones de salir aquí

            # Si no hay manos presentes sobre los botones, restablecer los estados presionados
            if not replay_hand_present:
                replay_pressed = False
            if not exit_hand_present:
                exit_pressed = False

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

while True:
    game_loop()
    if not show_final_screen():
        end_game.play()
        break


cap.release()
cv2.destroyAllWindows()

