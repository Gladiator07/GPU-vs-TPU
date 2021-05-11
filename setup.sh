
echo "Installing dependencies"
pip3 install --upgrade --force-reinstall --no-deps kaggle

# Put your Kaggle api key path here
echo "Fetching your Kaggle API Key"
kaggle_api_key_path='/content/drive/MyDrive/Kaggle/kaggle.json'

# This snippet will install kaggle api and connect your api-key to it
mkdir -p ~/.kaggle
echo "Setting up your Kaggle key to API..."
cp $kaggle_api_key_path ~/.kaggle/
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"

cd /content/GPU-vs-TPU
mkdir input/
cd input/
kaggle competitions download -c tpu-getting-started
unzip tpu-getting-started.zip

cd ..
mkdir encoded_data/
cd src/
python3 preprocess.py

cd ..
