import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src 目录
root_dir = parent_dir.parent     # 项目根目录
sys.path.append(str(root_dir))

def run_tests():
    """运行所有测试"""
    # 获取测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行所有以test_开头的Python文件
    test_files = [
        f for f in os.listdir(test_dir) 
        if f.startswith("test_") and f.endswith(".py")
    ]
    
    # 打印测试文件
    print(f"发现 {len(test_files)} 个测试文件:")
    for i, file in enumerate(test_files, 1):
        print(f"{i}. {file}")
    
    # 运行测试
    print("\n开始运行测试...")
    exit_code = pytest.main(["-v", test_dir])
    
    # 返回测试结果
    return exit_code

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
