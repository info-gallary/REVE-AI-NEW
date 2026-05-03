import cv2

# Try device index 0, 1, 2,... to find your USB microscope
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Change index if needed

if not cap.isOpened():
    print("❌ Could not open USB microscope.")
    exit()

print("🔍 USB Microscope Live Feed - Press SPACE to capture, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("🔬 USB Microscope - Live Preview", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("🚪 Exiting...")
        break
    elif key % 256 == 32:  # SPACE
        cv2.imwrite("usb_microscope_capture.jpg", frame)
        print("✅ Image saved as usb_microscope_capture.jpg")
        break

cap.release()
cv2.destroyAllWindows()
