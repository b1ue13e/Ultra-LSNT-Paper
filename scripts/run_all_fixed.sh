#!/bin/bash
# 补全所有未运行/出错/部分运行的实验脚本
# 在后台一次性运行所有实验，生成完整真实结果
# RTX 4090D 24GB 优化版

# 启用严格错误处理（移除 -e 以允许单个脚本失败后继续运行）
set -u -o pipefail

# 配置
MIN_DISK_SPACE_MB=5000  # 需要至少5GB磁盘空间
MAX_GPU_MEM_MB=23000    # RTX 4090D 显存阈值
LOG_RETENTION_DAYS=7    # 日志保留天数

echo "================================================"
echo "Ultra-LSNT 实验补全脚本 (run_all.sh) - RTX 4090D 优化版"
echo "开始时间: $(date)"
echo "系统信息: PyTorch 2.1.2, CUDA 11.8, RTX 4090D 24GB"
echo "================================================"

# 硬件检测函数
detect_hardware() {
    echo "硬件检测..."
    
    # CPU检测
    CPU_CORES=$(nproc --all 2>/dev/null || echo 1)
    echo "  CPU核心数: $CPU_CORES"
    
    # 内存检测
    if [ -f /proc/meminfo ]; then
        TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024))
        echo "  系统内存: ${TOTAL_MEM_GB}GB"
    fi
    
    # GPU检测
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 2>/dev/null || echo "0")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "Unknown")
        
        if [ "$GPU_MEM" != "0" ] && [ ! -z "$GPU_MEM" ]; then
            echo "  GPU型号: $GPU_NAME"
            echo "  GPU显存: ${GPU_MEM}MB"
            
            # 根据显存设置优化参数（仅设置标志，不导出具体参数以避免环境变量污染）
            if [ "$GPU_MEM" -ge "$MAX_GPU_MEM_MB" ]; then
                export GPU_OPTIMIZED=1
                echo "  检测到RTX 4090D级别GPU，启用GPU优化模式"
                echo "  注意：具体参数（BATCH_SIZE等）由各脚本的--gpu_optimized参数控制"
            else
                echo "  使用默认配置（GPU显存不足）"
                export GPU_OPTIMIZED=0
            fi
        else
            echo "  GPU检测失败，使用CPU模式"
            export GPU_OPTIMIZED=0
        fi
    else
        echo "  未检测到NVIDIA GPU，使用CPU模式"
        export GPU_OPTIMIZED=0
    fi
    
    # CUDA版本检测
    if python3 -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo "  警告: 未检测到PyTorch，实验可能失败"
        PYTHON_CMD="python3"
    fi
    export PYTHON_CMD
    echo "  使用Python命令: $PYTHON_CMD"
}

# 磁盘空间检查
check_disk_space() {
    echo "磁盘空间检查..."
    
    # 检查当前目录空间
    if command -v df >/dev/null 2>&1; then
        AVAILABLE=$(df -m . 2>/dev/null | awk 'NR==2 {print $4}')
        if [ ! -z "$AVAILABLE" ]; then
            if [ "$AVAILABLE" -lt "$MIN_DISK_SPACE_MB" ]; then
                echo "  错误: 磁盘空间不足，需要至少${MIN_DISK_SPACE_MB}MB，当前${AVAILABLE}MB"
                echo "  建议: 清理临时文件或使用更大磁盘"
                return 1
            else
                echo "  磁盘空间充足: ${AVAILABLE}MB 可用"
                return 0
            fi
        else
            echo "  警告: 无法获取磁盘空间信息"
            return 0
        fi
    else
        echo "  警告: 无法检查磁盘空间，跳过检查"
        return 0
    fi
}

# 清理旧日志
clean_old_logs() {
    echo "清理旧日志（保留${LOG_RETENTION_DAYS}天）..."
    if [ -d "logs" ]; then
        find logs -name "*.log" -type f -mtime +${LOG_RETENTION_DAYS} -delete 2>/dev/null || true
        echo "  日志清理完成"
    fi
}

# 资源监控
monitor_resources() {
    echo "=== 资源监控 ==="
    
    # 内存使用
    if [ -f /proc/meminfo ]; then
        FREE_MEM=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}')
        if [ ! -z "$FREE_MEM" ]; then
            FREE_MEM_MB=$((FREE_MEM / 1024))
            echo "  可用内存: ${FREE_MEM_MB}MB"
        fi
    fi
    
    # GPU使用
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 2>/dev/null || echo "N/A")
        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 2>/dev/null || echo "N/A")
        echo "  GPU使用率: $GPU_UTIL"
        echo "  GPU显存使用: $GPU_MEM_USED"
    fi
    
    # 磁盘使用
    if command -v df >/dev/null 2>&1; then
        DISK_USE=$(df -h . 2>/dev/null | awk 'NR==2 {print $5}' || echo "N/A")
        echo "  磁盘使用率: $DISK_USE"
    fi
    
    echo "================="
}

# 检查必需的数据文件
check_required_files() {
    echo "检查必需的数据文件..."
    
    REQUIRED_FILES=(
        "wind_final.csv"
        "ultra_lsnt_timeseries.py"
        "run_ablation_study.py"
    )
    
    missing_files=0
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "  ✅ $file 存在"
        else
            echo "  ❌ $file 不存在"
            missing_files=$((missing_files + 1))
        fi
    done
    
    if [ $missing_files -gt 0 ]; then
        echo "警告: 缺少 $missing_files 个必需文件，部分实验可能失败"
        return 1
    fi
    return 0
}

# 函数：检查脚本是否支持 --gpu_optimized 参数
script_supports_gpu_optimized() {
    local script_name=$1
    # 支持 --gpu_optimized 参数的脚本列表
    case "$script_name" in
        "ultra_lsnt_timeseries.py"|"run_sota.py")
            return 0  # 支持
            ;;
        *)
            return 1  # 不支持
            ;;
    esac
}

# 函数：运行脚本并记录日志
run_script() {
    local script_name=$1
    local log_file="logs/${script_name%.py}_${TIMESTAMP}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行: $script_name" | tee -a "$MAIN_LOG"
    
    # 监控资源（每10个脚本监控一次）
    if [ $((SCRIPT_COUNT % 10)) -eq 0 ]; then
        monitor_resources | tee -a "$MAIN_LOG"
    fi
    
    if [ -f "$script_name" ]; then
        # 检查是否是Python脚本
        if [[ "$script_name" == *.py ]]; then
            # 传递GPU优化参数给脚本（仅当脚本支持且GPU优化启用时）
            if [ "$GPU_OPTIMIZED" -eq 1 ] && script_supports_gpu_optimized "$script_name"; then
                echo "  使用GPU优化参数运行" | tee -a "$MAIN_LOG"
                $PYTHON_CMD "$script_name" --gpu_optimized 2>&1 | tee "$log_file"
            else
                $PYTHON_CMD "$script_name" 2>&1 | tee "$log_file"
            fi
            exit_code=${PIPESTATUS[0]}
        else
            echo "错误: $script_name 不是Python脚本" | tee -a "$MAIN_LOG"
            exit_code=1
        fi
    else
        echo "错误: 文件 $script_name 不存在" | tee -a "$MAIN_LOG"
        exit_code=1
    fi
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 成功: $script_name" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 失败: $script_name (退出码: $exit_code)" | tee -a "$MAIN_LOG"
        # 记录失败原因
        if [ -f "$log_file" ]; then
            echo "  失败日志片段:" | tee -a "$MAIN_LOG"
            tail -10 "$log_file" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        fi
        return 1
    fi
}

# 检查关键实验是否完成
check_key_experiment() {
    local exp_name=$1
    local script_pattern=$2
    
    if grep -q "✅ 成功.*$script_pattern" "$MAIN_LOG"; then
        echo "  ✅ $exp_name: 已完成"
    elif grep -q "❌ 失败.*$script_pattern" "$MAIN_LOG"; then
        echo "  ❌ $exp_name: 运行失败"
    else
        echo "  ⚠️  $exp_name: 未运行"
    fi
}

# ========== 主程序开始 ==========

# 执行硬件检测
detect_hardware

# 执行磁盘空间检查
if ! check_disk_space; then
    echo "磁盘空间检查失败，是否继续？(y/N)"
    read -r confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "用户取消执行"
        exit 1
    fi
    echo "用户确认继续执行"
fi

# 检查必需文件
check_required_files

# 清理旧日志
clean_old_logs

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/run_all_${TIMESTAMP}.log"

# 记录硬件信息
echo "硬件配置信息:" | tee -a "$MAIN_LOG"
detect_hardware | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 初始资源监控
monitor_resources | tee -a "$MAIN_LOG"

SCRIPT_COUNT=0

# 第一步：修复和运行出错的脚本
echo "第一步：修复和运行出错的脚本" | tee -a "$MAIN_LOG"

# 1.1 消融研究 (之前运行失败)
echo "1.1 重新运行消融研究 (修复CUDA错误)..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_ablation_study.py"

# 1.2 完成SOTA模型对比
echo "1.2 完成SOTA模型对比 (补全Autoformer)..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_sota.py"

# 第二步：运行完全未运行的效率测试脚本
echo "第二步：运行效率测试脚本" | tee -a "$MAIN_LOG"

# 2.1 效率分析（根据GPU优化调整参数）
echo "2.1 运行效率测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
if [ "$GPU_OPTIMIZED" -eq 1 ]; then
    echo "  使用GPU优化参数运行效率测试" | tee -a "$MAIN_LOG"
fi
run_script "test_efficiency.py"

# 2.2 MAPE测试
echo "2.2 运行MAPE测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "test_mape.py"

# 2.3 概率预测测试
echo "2.3 运行概率预测测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "test_probabilistic.py"

# 2.4 鲁棒性修复测试
echo "2.4 运行鲁棒性修复测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "test_robustness_fix.py"

# 第三步：运行统计显著性测试
echo "第三步：运行统计显著性测试" | tee -a "$MAIN_LOG"

# 3.1 Diebold-Mariano检验
echo "3.1 运行Diebold-Mariano检验..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_dm_test.py"

# 3.2 统计显著性测试
echo "3.2 运行统计显著性测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_significance.py"

# 3.3 通用显著性测试
echo "3.3 运行通用显著性测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_universal_significance.py"

# 第四步：运行鲁棒性对比实验
echo "第四步：运行鲁棒性对比实验" | tee -a "$MAIN_LOG"

# 4.1 GBDT鲁棒性对比
echo "4.1 运行GBDT鲁棒性对比..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_gbdt_robustness.py"

# 4.2 真实噪声对战
echo "4.2 运行真实噪声对战..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_real_noise_battle.py"

# 4.3 真实实验 (可能重复但运行确认)
echo "4.3 运行真实实验..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_real_experiments.py"

# 第五步：运行消融研究扩展
echo "第五步：运行消融研究扩展" | tee -a "$MAIN_LOG"

# 5.1 通用消融研究
echo "5.1 运行通用消融研究..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_universal_ablation.py"

# 5.2 DLinear修复版本
echo "5.2 运行DLinear修复版本..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_dlinear_fixed.py"

# 第六步：运行分析和可视化脚本
echo "第六步：运行分析和可视化脚本" | tee -a "$MAIN_LOG"

# 6.1 生成所有图表
echo "6.1 生成所有图表..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_all_figures.py"

# 6.2 经典基线模型
echo "6.2 运行经典基线模型..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "classical_baselines.py"

# 6.3 分层分析
echo "6.3 运行分层分析..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "stratified_analysis.py"

# 6.4 调度指标对比
echo "6.4 运行调度指标对比..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "compare_dispatch_metrics.py"

# 第七步：重新运行部分成功的脚本以确保完整性
echo "第七步：重新运行部分成功的脚本以确保完整性" | tee -a "$MAIN_LOG"

# 7.1 通用鲁棒性测试 (之前state_dict不匹配)
echo "7.1 重新运行通用鲁棒性测试..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
run_script "run_universal_robustness.py"

# 7.2 GBDT脚本 (运行fixed版本)
echo "7.2 运行GBDT脚本 (fixed版本)..." | tee -a "$MAIN_LOG"
SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
if [ -f "run_gbdt.py" ]; then
    # 检查是否有fixed版本
    if [ -f "run_gbdt_fixed.py" ]; then
        run_script "run_gbdt_fixed.py"
    else
        run_script "run_gbdt.py"
    fi
fi

# 完成总结
echo "================================================"
echo "实验补全完成!" | tee -a "$MAIN_LOG"
echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
echo "主日志文件: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "================================================"

# 最终资源监控
echo "最终资源状态:" | tee -a "$MAIN_LOG"
monitor_resources | tee -a "$MAIN_LOG"

# 生成汇总报告
echo "生成实验执行汇总报告..." | tee -a "$MAIN_LOG"
echo "实验脚本总数: $(grep -c "开始运行:" "$MAIN_LOG")" | tee -a "$MAIN_LOG"
echo "成功脚本数: $(grep -c "✅ 成功:" "$MAIN_LOG")" | tee -a "$MAIN_LOG"
echo "失败脚本数: $(grep -c "❌ 失败:" "$MAIN_LOG")" | tee -a "$MAIN_LOG"

# 检查关键实验是否完成
echo "" | tee -a "$MAIN_LOG"
echo "关键实验状态检查:" | tee -a "$MAIN_LOG"

check_key_experiment "消融研究" "run_ablation_study"
check_key_experiment "效率分析" "test_efficiency"
check_key_experiment "统计显著性检验" "run_dm_test"
check_key_experiment "GBDT鲁棒性对比" "run_gbdt_robustness"
check_key_experiment "SOTA模型对比" "run_sota"
check_key_experiment "通用鲁棒性测试" "run_universal_robustness"

echo "" | tee -a "$MAIN_LOG"
echo "所有实验日志保存在 logs/ 目录中" | tee -a "$MAIN_LOG"
echo "使用命令查看详细日志: tail -f $MAIN_LOG" | tee -a "$MAIN_LOG"

# 磁盘空间最终检查
echo "" | tee -a "$MAIN_LOG"
echo "磁盘空间最终检查:" | tee -a "$MAIN_LOG"
if command -v df >/dev/null 2>&1; then
    AVAILABLE=$(df -m . 2>/dev/null | awk 'NR==2 {print $4}')
    if [ ! -z "$AVAILABLE" ]; then
        echo "剩余磁盘空间: ${AVAILABLE}MB"
        if [ "$AVAILABLE" -lt 1000 ]; then
            echo "警告: 磁盘空间低于1GB，建议清理"
        fi
    fi
fi

echo "================================================"
echo "实验完成总结:"
echo "  总脚本数: $SCRIPT_COUNT"
echo "  开始时间: $(grep "开始时间:" "$MAIN_LOG" | head -1)"
echo "  结束时间: $(date)"
echo "  日志文件: $MAIN_LOG"
echo "================================================"

# 如果有关键实验失败，返回非零退出码
FAILED_COUNT=$(grep -c "❌ 失败:" "$MAIN_LOG")
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "警告: 有 $FAILED_COUNT 个脚本失败，请检查日志"
    exit 1
else
    echo "所有实验成功完成!"
    exit 0
fi