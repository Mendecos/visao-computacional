import cv2
import numpy as np

def load_yolo_model(config_path, weights_path, names_path):
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    with open(names_path, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    return model, class_names

def get_output_layers(model):
    layer_names = model.getLayerNames()
    return [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

def process_frame(frame, model, output_layers, class_names):
    height, width, _ = frame.shape


    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    predictions = model.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for prediction in predictions:
        for detection in prediction:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    return indices, boxes, confidences, class_ids

def detect_only_birds(video_source, model, output_layers, class_names):
    capture = cv2.VideoCapture(video_source)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        indices, boxes, confidences, class_ids = process_frame(frame, model, output_layers, class_names)

        for i in indices:
            i = i[0] 
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            confidence = confidences[i]

            # Mostrar apenas detecções da classe "bird"
            if label == "bird":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  
        cv2.imshow("Detecção de Pássaros", frame)

        # Sair ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


config_file = 'yolov3-tiny.cfg'
weights_file = 'yolov3-tiny.weights'
names_file = 'coco.names'

yolo_model, yolo_classes = load_yolo_model(config_file, weights_file, names_file)
yolo_output_layers = get_output_layers(yolo_model)

video_file = 'detec-passaro\\passaro.mp4'
detect_only_birds(video_file, yolo_model, yolo_output_layers, yolo_classes)
