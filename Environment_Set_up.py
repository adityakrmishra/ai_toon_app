# Install Python (if not already installed)
sudo apt-get update
sudo apt-get install python3.8
sudo apt-get install python3.8-venv python3.8-dev

# Install pip (Python package manager)
sudo apt-get install python3-pip

# Install Anaconda (optional, for managing environments and packages)
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-Linux-x86_64.sh
bash Anaconda3-2023.07-Linux-x86_64.sh
source ~/.bashrc

# Create a virtual environment using venv
python3 -m venv myenv
source myenv/bin/activate

# Create a virtual environment using conda (if Anaconda is installed)
conda create --name myenv
conda activate myenv

# Install necessary libraries
pip install tensorflow opencv-python matplotlib

# Install Jupyter Notebook
pip install jupyter

# Install Visual Studio Code (VS Code)
sudo snap install --classic code

# Install Python extension for VS Code
code --install-extension ms-python.python

# Set up Git for version control
sudo apt-get install git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize a Git repository
git init
git add .
git commit -m "Initial commit"

# Test TensorFlow and OpenCV installation
echo "import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

print('TensorFlow version:', tf.__version__)
print('OpenCV version:', cv2.__version__)

hello = tf.constant('Hello, TensorFlow!')
tf.print(hello)

img = cv2.imread('path_to_image.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()" > test_setup.py

python test_setup.py
