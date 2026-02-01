# Simple Makefile for MyEngine

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -I.
LDFLAGS = -lGL -lpthread

# 源文件 (避免重复)
SOURCES = $(wildcard core/**/*.cpp core/*.cpp) \
          $(wildcard scene/2d/*.cpp scene/3d/*.cpp scene/main/*.cpp scene/resources/*.cpp scene/physics/*.cpp scene/particles/*.cpp scene/audio/*.cpp scene/animation/*.cpp scene/scripting/*.cpp) \
          $(wildcard renderer/**/*.cpp renderer/*.cpp) \
          $(wildcard servers/**/*.cpp servers/*.cpp) \
          $(wildcard platform/**/*.cpp platform/*.cpp) \
          main.cpp

HEADERS = $(wildcard core/**/*.h core/*.h) \
          $(wildcard scene/**/*.h scene/*.h) \
          $(wildcard scene/resources/*.h) \
          $(wildcard renderer/**/*.h renderer/*.h) \
          $(wildcard servers/**/*.h servers/*.h) \
          $(wildcard scripts/**/*.h scripts/*.h) \
          $(wildcard platform/**/*.h platform/*.h)

# 目标
TARGET = bin/my_engine
OBJDIR = obj

# 创建目录结构
$(shell mkdir -p $(OBJDIR)/core $(OBJDIR)/scene $(OBJDIR)/renderer $(OBJDIR)/servers $(OBJDIR)/scripts $(OBJDIR)/platform)

# 对象文件
OBJECTS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))

# 主规则
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p bin
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: %.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -rf $(OBJDIR) bin

# 运行
run: $(TARGET)
	./$(TARGET)

# 重新编译
rebuild: clean all

# 检查文件
check:
	@echo "Source files:"
	@$(foreach f,$(SOURCES),echo $(f);)
	@echo ""
	@echo "Header files:"
	@$(foreach f,$(HEADERS),echo $(f);)

.PHONY: all clean run rebuild check
