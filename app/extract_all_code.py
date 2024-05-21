import cv2
import pytesseract
import multiprocessing
from multiprocessing import Pool
import re

# ChatGPT
# python multiprocessing program to extract only programming code from video using opencv and tesseract ocr with limited memory saving frames into text file

# Set up pytesseract path (if required)
# For example, on Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# Function to check if the text is likely to be programming code
def is_code(text):
    code_pattern = re.compile(r"""
        (\b(if|else|while|for|return|int|float|double|char|void|import|from|class|def|print|include|main)\b|  # common keywords
        [\{\}\[\]\(\)<>;:=]|  # common symbols
        \b\d+\b|  # numbers
        [\w]+\.\w+|  # object properties or functions
        [#]\w+|  # comments in some languages
        ['"][\s\S]*['"])  # strings
    """, re.VERBOSE)

    return bool(code_pattern.search(text))


# Function to process each frame
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary)
    if any(is_code(line) for line in text.split('\n')):
        return frame
    else:
        return None


# Function to process frames in chunks
def process_chunk(frames):
    return [process_frame(frame) for frame in frames]


# Function to process video frames in chunks
def process_video(video_path, chunk_size=1000):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pool = Pool(processes=multiprocessing.cpu_count())
    chunk_frames = []

    for start in range(0, frame_count, chunk_size):
        frames = []
        counter = 0
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            if counter % 50 == 0:
                frames.append(frame)
            counter += 1

        if frames:
            chunk_result = pool.apply_async(process_chunk, (frames,))
            chunk_frames.append(chunk_result)

    cap.release()
    pool.close()
    pool.join()

    frames_with_code = [frame for result in chunk_frames for frame in result.get() if frame is not None]
    return frames_with_code


def extract_code_from_frame(frame):
    """
    Extracts code from a given frame using Tesseract OCR.
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(binary, config=custom_config)

    # Filter out non-code lines
    code_lines = [line.strip() for line in text.split('\n')]

    return '\n'.join(code_lines)

# Function to save frames containing code as images
def save_frames(frames, output_dir, output_file):
    unique_code = set()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(frames):
        cv2.imwrite(f"{output_dir}/frame_{i}.png", frame)
        # Extract code from the frame
        code = extract_code_from_frame(frame)
        if code:
            unique_code.add(code)

    # Write the unique extracted code to a text file
    with open(output_file, 'w') as f:
        for code in unique_code:
            f.write(code + '\n')


# Main function
if __name__ == '__main__':
    import os

    video_path = 'oop(1).mp4'
    output_dir = 'frames_with_code'
    output_file = 'extracted_code.txt'

    frames_with_code = process_video(video_path)

    save_frames(frames_with_code, output_dir, output_file)

    print(f"Code-containing frames extraction complete. Check '{output_dir}' for the output images. Check '{output_file}' for the output file.")
