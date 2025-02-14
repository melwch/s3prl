# Prerequisites

## Python dependency libraries
To install Python dependency libraries that this program needs, locate "requirement.txt" file and run either method:
- python3 -m pip install -r requirements.txt
- pip3 install -r requirements.txt

## FFMPEG
For this tool to work, you will need the full suite of FFMPEG encoders and decoders and strongly recommend to install from source. Follow the guide link to install ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu. 

Below is the incomplete ffmpeg installation receipe for Ubuntu 20.04 as there are too many dependency libraries to install:

mkdir ~/ffmpeg_sources

cd ~/ffmpeg_sources && wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2 && tar xjvf nasm-2.15.05.tar.bz2 && cd nasm-2.15.05 && ./autogen.sh && PATH="$HOME/bin:$PATH" ./configure --prefix="/usr" --bindir="/usr/bin" && make clean && make -j 8 && sudo make install && cd ..

cd ~/ffmpeg_sources && \
git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom && \
rm -rf aom/aom_build && mkdir -p aom/aom_build && \
cd aom/aom_build && cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/usr" -DBUILD_SHARED_LIBS=1 -DCMAKE_BUILD_TYPE=Release -DENABLE_NASM=on .. && make -j 8 && sudo make install

cd ~/ffmpeg_sources && git -C SVT-AV1 pull 2> /dev/null || git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
rm -rf SVT-AV1/build && mkdir -p SVT-AV1/build && cd SVT-AV1/build && \
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/usr" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && make -j 8 && sudo make install

cd ~/ffmpeg_sources && \
wget https://github.com/Netflix/vmaf/archive/v2.1.1.tar.gz && \
tar xvf v2.1.1.tar.gz && \
rm -rf vmaf-2.1.1/libvmaf/build && mkdir -p vmaf-2.1.1/libvmaf/build &&\
cd vmaf-2.1.1/libvmaf/build && \
meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static .. --prefix "/usr" --bindir="/usr/bin" --libdir="/usr/lib" && ninja && sudo ninja install

cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && mkdir build && cd build && ../configure --prefix="/usr" --pkg-config-flags="--static" --extra-cflags="-I/usr/include" --extra-ldflags="-L/usr/lib" --extra-libs="-lpthread -lm" --extra-version=1ubuntu0.1 --toolchain=hardened --ld="g++" --bindir="/usr/bin" --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-libfdk-aac --enable-libsvtav1 --enable-libdav1d --enable-nonfree --disable-stripping --disable-filter=resample --enable-libopencore-amrnb --enable-version3 --enable-avisynth --enable-gpl --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-nvenc --enable-chromaprint --enable-frei0r --enable-libx264 && make clean && make -j 8 && sudo make install

## FFMPEG-NORMALIZE
This is a python library that allows user to configure the loudness normalization settings and generate the FFMPEG command for execution
To install, use either method:
- python3 -m pip install ffmpeg-normalize
- pip3 install ffmpeg-normalize
