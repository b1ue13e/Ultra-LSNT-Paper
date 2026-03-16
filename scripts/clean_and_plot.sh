#!/bin/bash
# clean_and_plot.sh - 清理所有图片并重新生成所有图表，确保无乱码
# 该脚本会：
# 1. 删除所有现有的图片文件
# 2. 确保所有绘图脚本使用正确的英文字体
# 3. 按顺序运行所有图片生成脚本
# 4. 验证生成的图片无乱码

echo "=============================================="
echo "开始清理并重新生成所有图表"
echo "日期: $(date)"
echo "=============================================="

# 切换到脚本所在目录
cd "$(dirname "$0")" || exit 1

# 函数：运行Python脚本并检查错误
run_python_script() {
    local script_name=$1
    local description=$2
    
    echo ""
    echo "执行: $description"
    echo "脚本: $script_name"
    echo "----------------------------------------------"
    
    if [ ! -f "$script_name" ]; then
        echo "警告: 找不到脚本 $script_name，跳过..."
        return 1
    fi
    
    # 运行脚本
    python3 "$script_name"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 成功: $script_name 完成"
        return 0
    else
        echo "❌ 失败: $script_name 退出码 $exit_code"
        return 1
    fi
}

# 步骤1: 删除所有现有图片文件
echo ""
echo "步骤1: 清理所有现有图片文件"
echo "----------------------------------------------"

# 定义要删除的图片文件扩展名
extensions=("*.png" "*.jpg" "*.jpeg" "*.pdf" "*.svg" "*.eps")

# 删除各个目录中的图片
for ext in "${extensions[@]}"; do
    # 当前目录
    if ls $ext 1>/dev/null 2>&1; then
        echo "删除当前目录中的 $ext 文件..."
        rm -f $ext
    fi
    
    # figures/ 目录
    if [ -d "figures" ]; then
        echo "删除 figures/ 目录中的 $ext 文件..."
        find figures -name "$ext" -type f -delete 2>/dev/null || true
    fi
    
    # images/ 目录
    if [ -d "images" ]; then
        echo "删除 images/ 目录中的 $ext 文件..."
        find images -name "$ext" -type f -delete 2>/dev/null || true
    fi
    
    # engineering_charts/ 目录
    if [ -d "engineering_charts" ]; then
        echo "删除 engineering_charts/ 目录中的 $ext 文件..."
        find engineering_charts -name "$ext" -type f -delete 2>/dev/null || true
    fi
done

echo "✅ 图片清理完成"

# 步骤2: 创建必要目录
echo ""
echo "步骤2: 创建必要的目录结构"
echo "----------------------------------------------"
mkdir -p figures
mkdir -p images
mkdir -p engineering_charts
mkdir -p figures/ablation
mkdir -p figures/architecture
mkdir -p figures/computational_efficiency
mkdir -p figures/economic
mkdir -p figures/dispatch
mkdir -p figures/domain_generalization
mkdir -p figures/expert_analysis
mkdir -p figures/robustness

echo "✅ 目录创建完成"

# 步骤3: 确保字体设置正确
echo ""
echo "步骤3: 配置英文字体设置"
echo "----------------------------------------------"
if [ -f "font_setup_final.py" ]; then
    echo "✅ 找到字体设置文件: font_setup_final.py"
    # 导入字体设置以确保所有脚本使用正确的字体
    python3 -c "
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
print('字体配置已应用: sans-serif (DejaVu Sans, Arial, Helvetica)')
"
else
    echo "⚠️  警告: 找不到 font_setup_final.py，创建默认字体设置..."
    cat > font_setup_final.py << 'EOF'
#!/usr/bin/env python3
# 通用字体设置，避免警告
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
EOF
    echo "✅ 已创建字体设置文件"
fi

# 步骤4: 运行所有图片生成脚本（按逻辑顺序）
echo ""
echo "步骤4: 运行所有图片生成脚本"
echo "=============================================="

# 创建成功和失败列表
success_list=()
fail_list=()

# 1. 基础性能对比
run_python_script "plot_results.py" "基础性能对比图表"
if [ $? -eq 0 ]; then success_list+=("plot_results.py"); else fail_list+=("plot_results.py"); fi

# 2. 调度对比图表
run_python_script "plot_dispatch_comparison.py" "调度对比图表"
if [ $? -eq 0 ]; then success_list+=("plot_dispatch_comparison.py"); else fail_list+=("plot_dispatch_comparison.py"); fi

# 3. 工程图表
run_python_script "engineering_charts.py" "工程对比图表"
if [ $? -eq 0 ]; then success_list+=("engineering_charts.py"); else fail_list+=("engineering_charts.py"); fi

# 4. 三维散点图
run_python_script "plot_3d_scatter.py" "三维性能散点图"
if [ $? -eq 0 ]; then success_list+=("plot_3d_scatter.py"); else fail_list+=("plot_3d_scatter.py"); fi

# 5. 架构图
run_python_script "plot_arch.py" "模型架构图"
if [ $? -eq 0 ]; then success_list+=("plot_arch.py"); else fail_list+=("plot_arch.py"); fi

# 6. 论文专用图
run_python_script "plot_paper_figs.py" "论文专用图表"
if [ $? -eq 0 ]; then success_list+=("plot_paper_figs.py"); else fail_list+=("plot_paper_figs.py"); fi

# 7. 效率与专家分析
run_python_script "test_efficiency.py" "效率边界对比"
if [ $? -eq 0 ]; then success_list+=("test_efficiency.py"); else fail_list+=("test_efficiency.py"); fi

run_python_script "plot_expert_heatmap.py" "专家激活热力图"
if [ $? -eq 0 ]; then success_list+=("plot_expert_heatmap.py"); else fail_list+=("plot_expert_heatmap.py"); fi

run_python_script "plot_expert_heatmap_enhanced.py" "增强版专家对比热力图"
if [ $? -eq 0 ]; then success_list+=("plot_expert_heatmap_enhanced.py"); else fail_list+=("plot_expert_heatmap_enhanced.py"); fi

# 8. 鲁棒性对战图
run_python_script "battle_dlinear.py" "DLinear鲁棒性对战"
if [ $? -eq 0 ]; then success_list+=("battle_dlinear.py"); else fail_list+=("battle_dlinear.py"); fi

run_python_script "compare_robustness_real.py" "Transformer鲁棒性对比"
if [ $? -eq 0 ]; then success_list+=("compare_robustness_real.py"); else fail_list+=("compare_robustness_real.py"); fi

run_python_script "run_gbdt_robustness.py" "LightGBM鲁棒性对战"
if [ $? -eq 0 ]; then success_list+=("run_gbdt_robustness.py"); else fail_list+=("run_gbdt_robustness.py"); fi

run_python_script "battle_economics.py" "经济影响分析"
if [ $? -eq 0 ]; then success_list+=("battle_economics.py"); else fail_list+=("battle_economics.py"); fi

run_python_script "plot_dlinear_victory.py" "DLinear崩溃可视化"
if [ $? -eq 0 ]; then success_list+=("plot_dlinear_victory.py"); else fail_list+=("plot_dlinear_victory.py"); fi

run_python_script "real_robustness.py" "真实数据鲁棒性分析"
if [ $? -eq 0 ]; then success_list+=("real_robustness.py"); else fail_list+=("real_robustness.py"); fi

run_python_script "run_universal_robustness.py" "跨领域鲁棒性分析"
if [ $? -eq 0 ]; then success_list+=("run_universal_robustness.py"); else fail_list+=("run_universal_robustness.py"); fi

# 9. 多领域和高级图表
run_python_script "plot_multi_domain.py" "多领域性能对比"
if [ $? -eq 0 ]; then success_list+=("plot_multi_domain.py"); else fail_list+=("plot_multi_domain.py"); fi

run_python_script "generate_figures.py" "Applied Energy风格图表"
if [ $? -eq 0 ]; then success_list+=("generate_figures.py"); else fail_list+=("generate_figures.py"); fi

run_python_script "assemble_figures.py" "论文最终图表组装"
if [ $? -eq 0 ]; then success_list+=("assemble_figures.py"); else fail_list+=("assemble_figures.py"); fi

# 10. 完整流水线（如果存在）
if [ -f "run_all_figures.py" ]; then
    run_python_script "run_all_figures.py" "完整图表生成流水线"
    if [ $? -eq 0 ]; then success_list+=("run_all_figures.py"); else fail_list+=("run_all_figures.py"); fi
fi

# 步骤5: 验证生成的图片
echo ""
echo "步骤5: 验证生成的图片"
echo "=============================================="

# 统计生成的图片数量
echo "生成的图片统计:"
echo "----------------------------------------------"
figures_count=$(find . -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" | wc -l)
echo "总图片文件数: $figures_count"

if [ $figures_count -gt 0 ]; then
    echo ""
    echo "主要图片文件:"
    echo "----------------------------------------------"
    
    # 列出重要图片
    important_images=(
        "figures/performance_comparison.png"
        "dispatch_comparison_charts.png"
        "dispatch_detailed_analysis.png"
        "engineering_charts/overall_performance.png"
        "3d_performance_scatter.png"
        "fig_architecture.png"
        "fig_comparison.png"
        "fig_ablation.png"
        "fig_efficiency.png"
        "fig_real_expert_heatmap.png"
        "fig_expert_comparison_latest.png"
        "fig_battle_dlinear_victory.png"
        "fig_economic_impact.png"
        "fig_robustness_compare.png"
        "fig_gbdt_battle.png"
        "fig_universal_robustness.png"
        "fig_multi_domain.png"
        "Figure_2_Final.png"
        "Figure_3_Final.png"
        "Figure_4_Final.png"
        "Figure_5_Final.png"
    )
    
    for img in "${important_images[@]}"; do
        if [ -f "$img" ]; then
            echo "✅ $img"
        else
            echo "❌ $img (未找到)"
        fi
    done
    
    echo ""
    echo "图片文件大小检查:"
    echo "----------------------------------------------"
    # 检查是否有空文件或异常小的文件
    find . -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" -type f -size -1k 2>/dev/null | while read -r file; do
        echo "⚠️  警告: 文件可能为空或异常小: $file ($(stat -c%s "$file") 字节)"
    done
else
    echo "❌ 错误: 没有生成任何图片文件"
fi

# 步骤6: 生成执行报告
echo ""
echo "步骤6: 生成执行报告"
echo "=============================================="

report_file="plot_generation_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "图表生成报告"
    echo "================"
    echo "生成时间: $(date)"
    echo "总脚本数: $((${#success_list[@]} + ${#fail_list[@]}))"
    echo "成功脚本数: ${#success_list[@]}"
    echo "失败脚本数: ${#fail_list[@]}"
    echo ""
    echo "成功执行的脚本:"
    for script in "${success_list[@]}"; do
        echo "  ✅ $script"
    done
    echo ""
    if [ ${#fail_list[@]} -gt 0 ]; then
        echo "失败的脚本:"
        for script in "${fail_list[@]}"; do
            echo "  ❌ $script"
        done
    else
        echo "所有脚本都成功执行!"
    fi
    echo ""
    echo "生成的图片统计:"
    echo "总图片文件数: $figures_count"
    echo ""
    echo "重要图片状态:"
    for img in "${important_images[@]}"; do
        if [ -f "$img" ]; then
            echo "  ✅ $img"
        else
            echo "  ❌ $img (缺失)"
        fi
    done
} > "$report_file"

echo "✅ 执行报告已保存: $report_file"

# 最终总结
echo ""
echo "=============================================="
echo "图表生成任务完成"
echo "=============================================="
echo "成功脚本: ${#success_list[@]}"
echo "失败脚本: ${#fail_list[@]}"
echo "生成图片: $figures_count 个文件"
echo "报告文件: $report_file"
echo ""
if [ ${#fail_list[@]} -eq 0 ]; then
    echo "🎉 所有图表已成功生成！"
else
    echo "⚠️  有 ${#fail_list[@]} 个脚本失败，请检查日志。"
fi
echo "=============================================="