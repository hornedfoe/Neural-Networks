import cv2
import os

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))


def create_video(images_folder, output_file, frame_rate=24):
    images = [img for img in os.listdir(images_folder) if img.endswith(".png")]
    images.sort(key=extract_number)

    # Determine the size of the first image
    img_path = os.path.join(images_folder, images[0])
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # Write images to video
    for image in images:
        img_path = os.path.join(images_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer
    video.release()

    print(f"Video created: {output_file}")

# Example usage


create_video('PYTHON/images', 'evolution.mp4', frame_rate=8)
