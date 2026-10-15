#!/usr/bin/env bash
echo "Current working dir: $(pwd)"
PLATFORM=${1:-linux}
if [ "$PLATFORM" = "android" ]; then
    IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.7"
elif [ "$PLATFORM" = "linux" ]; then
    IMAGE="ghcr.io/snapdragon-toolchain/arm64-linux:v0.7"
else
    echo "Usage:  [linux|android]"; exit 1
fi
docker run -i -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 $IMAGE \
    sh -c "cd /workspace; ./scripts/build.sh $PLATFORM; echo done building!"
