# Download dependencies

# Download Huggy env:
wget "https://github.com/huggingface/Huggy/raw/main/Huggy.zip" -O ./Huggy.zip
unzip -d ./envs/ ./Huggy.zip
chmod -R 755 ./envs/Huggy/Huggy.x86_64
rm ./Huggy.zip

# Download unity's ml agents
# Clone the repository (can take 3min)
# It might requires to modify some dependencies for this to work
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
