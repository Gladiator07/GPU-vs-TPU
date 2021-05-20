
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
mkdir jpeg/
cd jpeg
kaggle datasets download -d msheriey/104-flowers-garden-of-eden
unzip 104-flowers-garden-of-eden
rm -rf *.zip
cd ..

# mkdir tfrecords/
# kaggle competitions download -c tpu-getting-started
# unzip tpu-getting-started.zip
# rm tpu-getting-started.zip
# cd ..


# cd ..
# mkdir encoded_data/
# cd src/
# python3 preprocess.py

cd ..

pip3 install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8.1-cp37-cp37m-linux_x86_64.whl
pip3 install wandb
pip3 install black