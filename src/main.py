import os

# print("🚀 Running Auto-Labeling...")
# os.system("python auto_label.py")

print("🚀 Training Model...")
os.system("python train.py")

print("🚀 Testing on New Images...")
os.system("python test.py")
