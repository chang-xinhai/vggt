#!/bin/bash

# 定义源目录（所有zip文件所在目录）和目标目录（解压后存放目录）
SRC_DIR="/root/autodl-tmp/vggt/GSO-Data-Utils/GSO_data/GSO"
DST_ROOT="/root/autodl-tmp/vggt/GSO-Data-Utils/GSO_data/GSO_unzip"

# 检查源目录是否存在
if [ ! -d "$SRC_DIR" ]; then
    echo "错误：源目录 $SRC_DIR 不存在！"
    exit 1
fi

# 创建目标根目录（不存在则创建，存在则跳过）
mkdir -p "$DST_ROOT"

# 统计zip文件数量
ZIP_COUNT=$(find "$SRC_DIR" -maxdepth 1 -type f -name "*.zip" | wc -l)
if [ "$ZIP_COUNT" -eq 0 ]; then
    echo "警告：源目录 $SRC_DIR 下没有找到 .zip 文件！"
    exit 0
fi

echo "找到 $ZIP_COUNT 个 zip 文件，开始解压..."
echo "源目录：$SRC_DIR"
echo "目标根目录：$DST_ROOT"
echo "----------------------------------------"

# 遍历源目录下的所有 .zip 文件（仅一级目录，不递归子文件夹）
find "$SRC_DIR" -maxdepth 1 -type f -name "*.zip" | while read -r ZIP_FILE; do
    # 获取 zip 文件名（不含路径、不含 .zip 后缀）
    ZIP_NAME=$(basename "$ZIP_FILE" .zip)
    # 每个 zip 对应的目标文件夹
    DST_DIR="$DST_ROOT/$ZIP_NAME"
    
    # 检查目标文件夹是否已存在
    if [ -d "$DST_DIR" ]; then
        echo "⚠️  目标文件夹 $DST_DIR 已存在，跳过解压（如需重新解压，请先删除该文件夹）"
        continue
    fi
    
    # 创建当前zip的目标文件夹
    mkdir -p "$DST_DIR" || {
        echo "❌ 创建文件夹 $DST_DIR 失败，跳过该文件"
        continue
    }
    
    # 解压 zip 文件到目标文件夹（-q 静默模式，-o 覆盖已存在文件，-d 指定目标目录）
    echo "正在解压：$ZIP_FILE -> $DST_DIR"
    unzip -q -o "$ZIP_FILE" -d "$DST_DIR"
    
    # 检查解压是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 解压成功：$ZIP_NAME"
    else
        echo "❌ 解压失败：$ZIP_FILE"
        # 解压失败则删除空文件夹
        rm -rf "$DST_DIR"
    fi
    echo "----------------------------------------"
done

echo "🎉 所有 zip 文件处理完成！"
echo "解压后文件存放于：$DST_ROOT"