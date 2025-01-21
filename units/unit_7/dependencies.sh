# Download dependencies

# Download Unity ml agent envs:
wget --no-check-certificate -r "https://drive.usercontent.google.com/open?id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL&authuser=0" -O ./SoccerTwos.zip
unzip -d ./envs/ ./SoccerTwos

chmod -R 755 ./envs/SoccerTwos/SoccerTwos.x86_64

# rm ./SoccerTwos.zip

# Download unity's ml agents
# Clone the repository (can take 3min)
# It might requires to modify some dependencies for this to work
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
