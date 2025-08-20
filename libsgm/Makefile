# === 編譯器與標準 ===
CXX := g++
CXXSTD := -std=c++17
CXXFLAGS := -O3 $(CXXSTD)

# 打開 LibSGM 的 OpenCV 包裝介面

# === OpenCV（分開 cflags / libs）===
PKG_CFLAGS := $(shell pkg-config --cflags opencv4)
PKG_LIBS   := $(shell pkg-config --libs   opencv4)

# === 檔名 ===
TARGET := test_libsgm
SRC := test_libsgm.cpp /home/aibox/itri/cnpy/cnpy.cpp

# === 包含路徑 ===
INCLUDES := \
    -I/home/aibox/itri/libSGM/include \
    -I/usr/include/opencv4 \
    -I/home/aibox/itri/cnpy

# === 連結路徑與函式庫 ===
SGM_LIB_DIR := /home/aibox/itri/libSGM/build/src
CUDA_LIB_DIR := /usr/local/cuda/lib64

LDFLAGS := \
  -Wl,--whole-archive -L/home/aibox/itri/libSGM/build/src -lsgm -Wl,--no-whole-archive \
  -L/usr/local/cuda/lib64 -lcudart \
  $(PKG_LIBS) \
  -lz


# === 最終目標 ===
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PKG_CFLAGS) $^ -o $@ $(LDFLAGS)

# === 清除 ===
clean:
	rm -f $(TARGET)
