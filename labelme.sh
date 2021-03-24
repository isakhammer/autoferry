
# on Linux
xhost +
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 -v $(pwd):/root/workdir wkentaro/labelme
