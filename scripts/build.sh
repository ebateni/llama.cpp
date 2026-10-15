#export HEXAGON_SDK_ROOT=/local/mnt2/workspace2/llama.cpp/latest/hexagon/6.4.0.2
#export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/

PLATFORM=${1:-linux}

rm -rf build-snapdragon
rm -rf pkg-snapdragon
rm -f pkg-snapdragon_${PLATFORM}.zip
rm -f libggml-htp-v81.so

if [ "$PLATFORM" = "android" ]; then
    cmake --preset arm64-android-snapdragon-release -B build-snapdragon
elif [ "$PLATFORM" = "linux" ]; then
    cmake --preset arm64-linux-snapdragon-release -B build-snapdragon
else
    echo "Usage: $0 [linux|android]"
    exit 1
fi

cmake --build build-snapdragon -j 16
cmake --install build-snapdragon --prefix pkg-snapdragon
chmod 777 ./pkg-snapdragon/lib/libggml-htp-v81.so
python3 swiv_build_utility.py  -i ./pkg-snapdragon/lib/libggml-htp-v81.so -o libggml-htp-v81.so
cp libggml-htp-v81.so ./pkg-snapdragon/lib/libggml-htp-v81.so
zip -r pkg-snapdragon_${PLATFORM}.zip pkg-snapdragon
