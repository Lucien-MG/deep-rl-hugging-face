# Download dependencies

# Download Unity ml agent envs:
wget "https://github.com/huggingface/Snowball-Target/raw/main/SnowballTarget.zip" -O ./SnowballTarget.zip
unzip -d ./envs/ ./SnowballTarget.zip

wget "https://huggingface.co/spaces/unity/ML-Agents-Pyramids/resolve/main/Pyramids.zip" -O ./Pyramids.zip
unzip -d ./envs/ ./Pyramids.zip

chmod -R 755 ./envs/SnowballTarget/SnowballTarget.x86_64
chmod -R 755 ./envs/Pyramids/Pyramids.x86_64

rm ./SnowballTarget.zip
rm ./Pyramids.zip

# Download unity's ml agents
# Clone the repository (can take 3min)
# It might requires to modify some dependencies for this to work
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
