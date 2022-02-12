import cv2
import base64


def get_base64_image(image):
    """
    Convert image to base 64.
    """
    try:
        _, image_buffer = cv2.imencode(".jpg", image)
        image_str = base64.b64encode(image_buffer).decode("utf-8")
        return "data:image/jpeg;base64, {0}".format(image_str)
    except BaseException:
        return None


def reshape_image(img, max_size=800):
    if img is None or max(img.shape) < max_size:
        return img

    h, w, _ = img.shape
    if h > w:
        w = int(w / (h / max_size))
        h = int(max_size)
    else:
        h = int(h / (w / max_size))
        w = int(max_size)

    print("Resizing image. Original shape: (%d, %d). New shape: (%d,%d)" % (img.shape[0], img.shape[1], h, w))

    return cv2.resize(img, (w, h))