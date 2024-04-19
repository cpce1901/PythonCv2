import cv2
import mediapipe as mp
import numpy as np
import paho.mqtt.client as mqtt

MQTT_BROKER = "146.190.124.66"
MQTT_PORT = 1883
MQTT_USERNAME = "SCollege"
MQTT_PASSWORD = "cpce1901"
MQTT_ID = "appWeb"
TOPIC=['demo/energia/', 'demo/temperatura/', 'demo/valvula/']
TOPIC_SEND = ['hands/', 'color/']


def on_connect(client, userdata, flags, rc, properties):
    
    if rc == 0:
        print("Conectado")
        for i in TOPIC:
            client.subscribe(i)

def on_message(client, userdata, msg):
    message = msg.payload.decode()


# Conexión MQTT
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_ID)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()


X_Y_INI_INTENSITY = 150
X_Y_INI_COLOR = 100

SLIDER_WIDTH_INTENSITY = 100
SLIDER_WIDTH_COLOR = 100

OFFSET =50
color_pointer = (0, 0, 255)
color_pointer2 = (255, 0, 0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.05) as hands:


    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape

        frame = cv2.flip(frame, 1)

        aux_frame_width_intensity = width - X_Y_INI_INTENSITY * 2
        aux_frame_heigth_intensity = SLIDER_WIDTH_INTENSITY
        aux_frame_intensity = np.zeros(frame.shape, np.uint8)

        aux_frame_heigth_color = height - X_Y_INI_COLOR * 2
        aux_frame_width_color = SLIDER_WIDTH_COLOR
        aux_frame_color = np.zeros(frame.shape, np.uint8)
      

        rectangle_intensity = cv2.rectangle(aux_frame_intensity, (X_Y_INI_INTENSITY , X_Y_INI_INTENSITY), (X_Y_INI_INTENSITY + aux_frame_width_intensity, X_Y_INI_INTENSITY + aux_frame_heigth_intensity), (255, 0, 0), -1)
        
        rectangle_color = cv2.rectangle(aux_frame_color, (width - X_Y_INI_COLOR, X_Y_INI_COLOR),(width - X_Y_INI_COLOR + aux_frame_width_color, height - X_Y_INI_COLOR ),(0, 255, 0), -1)

        output= cv2.addWeighted(frame, 1, rectangle_intensity, 0.7, 0)
        output = cv2.addWeighted(output, 1, rectangle_color, 0.7, 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)


        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)

                # Verificar si el dedo está dentro del área del rectángulo
            if X_Y_INI_INTENSITY < x < X_Y_INI_INTENSITY + aux_frame_width_intensity and X_Y_INI_INTENSITY < y < X_Y_INI_INTENSITY + aux_frame_heigth_intensity:

                ligth = np.interp(x, (X_Y_INI_INTENSITY + OFFSET , X_Y_INI_INTENSITY + aux_frame_width_intensity - OFFSET), (0, 255))
                mqtt_client.publish(TOPIC_SEND[0], int(ligth), 1, retain=True)

                cv2.circle(output, (x, y), 10, color_pointer, 3)
                cv2.circle(output, (x, y), 5, color_pointer, -1)

            # Verificar si el dedo está dentro del área del rectángulo vertical
            if width - X_Y_INI_COLOR < x < width - X_Y_INI_COLOR + aux_frame_width_color and X_Y_INI_COLOR < y < X_Y_INI_COLOR + aux_frame_heigth_color:
                color = np.interp(y, (X_Y_INI_COLOR + OFFSET , height - X_Y_INI_COLOR + OFFSET), (0, 6))
                mqtt_client.publish(TOPIC_SEND[1], int(color), 1, retain=True)

                cv2.circle(output, (x, y), 10, color_pointer2, 3)
                cv2.circle(output, (x, y), 5, color_pointer2, -1)

        cv2.imshow('output', output)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

mqtt_client.loop_stop()
cap.release()
cv2.destroyAllWindows()
