import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt') 

def process_image(image_path, output_txt='keypoints_output.txt'):
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd: Nie można wczytać obrazu.")
        return
    
    results = model(image)

    with open(output_txt, 'w') as f:
        for result in results:
            annotated_image = result.plot() 

            keypoints = result.keypoints.xy.cpu().numpy() 

            for person_idx, person_keypoints in enumerate(keypoints):
                f.write(f"Person {person_idx + 1}:\n")
                for kp_idx, (x, y) in enumerate(person_keypoints):
                    f.write(f"Keypoint {kp_idx}: x={x:.2f}, y={y:.2f}\n")
                f.write("\n")

            cv2.imshow('YOLOv8 Pose Estimation', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite('output_image.jpg', annotated_image)
        print(f"Zapisano obraz wynikowy jako 'output_image.jpg'")
        print(f"Zapisano współrzędne punktów kluczowych do '{output_txt}'")

def process_video(video_path, output_txt='keypoints_video_output.txt'):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Błąd: Nie można wczytać wideo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with open(output_txt, 'w') as f:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for result in results:
                annotated_frame = result.plot()  
                
                keypoints = result.keypoints.xy.cpu().numpy()  
                f.write(f"Frame {frame_idx}:\n")
                for person_idx, person_keypoints in enumerate(keypoints):
                    f.write(f"  Person {person_idx + 1}:\n")
                    for kp_idx, (x, y) in enumerate(person_keypoints):
                        f.write(f"    Keypoint {kp_idx}: x={x:.2f}, y={y:.2f}\n")
                    f.write("\n")
                f.write("\n")

                out.write(annotated_frame)

                cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Zapisano wideo wynikowe jako 'output_video.mp4'")
    print(f"Zapisano współrzędne punktów kluczowych do '{output_txt}'")

if __name__ == "__main__":
    
    image_path = 'sample_image.jpg'  
    process_image(image_path)

    video_path = 'sample_video.mp4'  
    process_video(video_path)
