# === 編譯器與標準 ===
CXX := g++
CXXFLAGS := -O3 -std=c++11

# === 檔名 ===
TARGET := test_libsgm
SRC := test_libsgm.cpp /home/aibox/itri/cnpy/cnpy.cpp

# === 包含路徑 ===
INCLUDES := \
    -I/home/aibox/itri/libSGM/include \
    -I/usr/include/opencv4 \
	-I/home/aibox/itri/cnpy

# === 連結路徑與函式庫 ===
LDFLAGS := \
    -L/home/aibox/itri/libSGM/build/src -lsgm \
    -L/usr/local/cuda/lib64 -lcudart \
    `pkg-config --cflags --libs opencv4`

# === 最終目標 ===
all: $(TARGET)

$(TARGET): $(SRC) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) -lz
 
# === 清除 ===
clean:
	rm -f $(TARGET)
