docker rm -f sam6d
DIR=$(pwd)/
xhost +local:1000 && docker run --name sam6d --gpus all -it -v /home:/home lihualiu/sam-6d:1.0 bash -c "cd $DIR && bash" -e DISPLAY="$DISPLAY" \
-v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    --network=host \
    --ipc=host