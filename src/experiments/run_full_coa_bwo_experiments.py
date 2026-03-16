#!/usr/bin/env python3
"""
COA/BWO 完整实验运行脚本

本脚本自动化运行所有COA/BWO实验，包括：
1. COA-BiLSTM 在四个数据集上的优化实验
2. BWO-SVR 在四个数据集上的优化实验  
3. BWO-CNN 在四个数据集上的优化实验

支持后台运行、日志记录和进度监控。
"""

import os
import sys
import json
import subprocess
import time
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import signal
import threading
import queue

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ExperimentRunner:
    """实验运行管理器"""
    
    def __init__(self, log_dir: str = "experiment_logs", 
                 results_dir: str = ".",
                 max_parallel: int = 1):
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.max_parallel = max_parallel
        
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # 实验配置
        self.experiments = [
            {
                "name": "coa_bilstm",
                "script": "coa_bilstm_experiment.py",
                "datasets": ["wind", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "description": "COA优化的BiLSTM模型"
            },
            {
                "name": "bwo_svr", 
                "script": "bwo_svr_experiment.py",
                "datasets": ["wind", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "description": "BWO优化的SVR模型"
            },
            {
                "name": "bwo_cnn",
                "script": "bwo_cnn_experiment.py", 
                "datasets": ["wind", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "description": "BWO优化的1D CNN模型"
            }
        ]
        
        # 状态跟踪
        self.status = {
            "total_experiments": 0,
            "completed": 0,
            "failed": 0,
            "running": 0,
            "queued": 0
        }
        
        self.start_time = None
        self.progress_file = self.log_dir / "experiment_progress.json"
        
    def setup_environment(self):
        """设置实验环境"""
        print("=" * 60)
        print("COA/BWO 完整实验运行器")
        print("=" * 60)
        print(f"日志目录: {self.log_dir}")
        print(f"结果目录: {self.results_dir}")
        print(f"最大并行度: {self.max_parallel}")
        
        # 验证必要文件
        required_files = [
            "coa_bilstm_experiment.py",
            "bwo_svr_experiment.py", 
            "bwo_cnn_experiment.py",
            "coa_algorithm.py",
            "bwo_algorithm.py",
            "coa_bwo_unified_split.py",
            "split_manifest_80_20_unified.json"
        ]
        
        missing_files = []
        for f in required_files:
            if not Path(f).exists():
                missing_files.append(f)
        
        if missing_files:
            print(f"\n❌ 缺少必要文件: {missing_files}")
            return False
        
        print("\n✅ 环境验证通过")
        return True
    
    def run_single_experiment(self, exp_name: str, script: str, 
                            dataset: str, seed: int, log_file: Path) -> bool:
        """运行单个实验"""
        cmd = [
            sys.executable, script,
            "--dataset", dataset,
            "--seed", str(seed),
            "--output", f"{exp_name}_{dataset}_{seed}.json"
        ]
        
        # 记录命令
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"开始时间: {datetime.datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
        
        try:
            # 运行实验
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时捕获输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    with open(log_file, 'a') as f:
                        f.write(output)
                    # 也输出到控制台（如果verbose）
                    if self.verbose:
                        print(output.rstrip())
            
            return_code = process.poll()
            
            with open(log_file, 'a') as f:
                f.write(f"\n退出码: {return_code}\n")
                f.write(f"结束时间: {datetime.datetime.now().isoformat()}\n")
            
            return return_code == 0
            
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"\n❌ 异常: {str(e)}\n")
            return False
    
    def update_split_manifest_for_wind_us(self):
        """更新划分清单以包含Wind US数据集"""
        manifest_path = Path("split_manifest_80_20_unified.json")
        if not manifest_path.exists():
            print("❌ 划分清单文件不存在")
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # 检查是否已包含wind_us
            datasets = [d["name"] for d in manifest["datasets"]]
            if "wind_us" in datasets:
                print("✅ Wind US已在划分清单中")
                return True
            
            # 添加wind_us数据集
            wind_us_config = {
                "name": "wind_us",
                "file": "wind_us.csv",
                "target_column": "power",
                "time_column": None,
                "total_samples": 105121,  # 从wc -l获得
                "training_set": {
                    "samples": 84096,  # 80%
                    "indices": "0:84096",
                    "ratio": 0.8,
                    "description": "First 84096 samples (80%)"
                },
                "test_set": {
                    "samples": 21025,  # 20%
                    "indices": "84096:105121",
                    "ratio": 0.2,
                    "description": "Last 21025 samples (20%)"
                },
                "split_validation": {
                    "no_overlap": True,
                    "chronological": True,
                    "consistent_across_runs": True
                }
            }
            
            # 更新原始wind配置为wind_cn
            for i, dataset in enumerate(manifest["datasets"]):
                if dataset["name"] == "wind":
                    manifest["datasets"][i]["name"] = "wind_cn"
                    manifest["datasets"][i]["description"] = "中国风电数据集"
                    break
            
            # 添加wind_us
            manifest["datasets"].append(wind_us_config)
            
            # 保存更新后的清单
            backup_path = Path("split_manifest_80_20_unified_backup.json")
            if not backup_path.exists():
                with open(backup_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print("✅ 已更新划分清单，包含Wind CN和Wind US")
            print(f"   数据集: {[d['name'] for d in manifest['datasets']]}")
            return True
            
        except Exception as e:
            print(f"❌ 更新划分清单失败: {e}")
            return False
    
    def run_all_experiments(self, verbose: bool = False, 
                          skip_completed: bool = True):
        """运行所有实验"""
        self.verbose = verbose
        self.start_time = datetime.datetime.now()
        
        print("\n" + "="*60)
        print("开始运行COA/BWO完整实验")
        print("="*60)
        
        # 更新数据集划分
        print("\n1. 更新数据集划分清单...")
        if not self.update_split_manifest_for_wind_us():
            print("⚠️  划分清单更新失败，继续使用原有配置")
        
        # 计算总实验数
        total = 0
        for exp in self.experiments:
            total += len(exp["datasets"]) * len(exp["seeds"])
        
        self.status["total_experiments"] = total
        self.status["queued"] = total
        
        print(f"\n2. 准备运行 {total} 个实验:")
        for exp in self.experiments:
            exp_count = len(exp["datasets"]) * len(exp["seeds"])
            print(f"   • {exp['description']}: {exp_count} 个实验")
        
        print(f"\n3. 开始实验运行 (最大并行度: {self.max_parallel})...")
        
        # 运行实验
        completed_experiments = []
        failed_experiments = []
        
        for exp_config in self.experiments:
            exp_name = exp_config["name"]
            script = exp_config["script"]
            
            print(f"\n{'='*40}")
            print(f"运行 {exp_config['description']}")
            print(f"{'='*40}")
            
            for dataset in exp_config["datasets"]:
                for seed in exp_config["seeds"]:
                    # 检查是否已存在结果
                    result_file = Path(f"{exp_name}_{dataset}_{seed}.json")
                    if skip_completed and result_file.exists():
                        print(f"   ✓ {exp_name}/{dataset}/seed{seed}: 已存在结果，跳过")
                        self.status["completed"] += 1
                        self.status["queued"] -= 1
                        completed_experiments.append({
                            "experiment": exp_name,
                            "dataset": dataset,
                            "seed": seed,
                            "status": "skipped"
                        })
                        continue
                    
                    # 创建日志文件
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file = self.log_dir / f"{exp_name}_{dataset}_{seed}_{timestamp}.log"
                    
                    print(f"   ▶ {exp_name}/{dataset}/seed{seed}: 运行中...")
                    
                    # 更新状态
                    self.status["running"] += 1
                    self.status["queued"] -= 1
                    
                    # 运行实验
                    success = self.run_single_experiment(
                        exp_name, script, dataset, seed, log_file
                    )
                    
                    # 更新状态
                    self.status["running"] -= 1
                    
                    if success:
                        self.status["completed"] += 1
                        print(f"   ✓ {exp_name}/{dataset}/seed{seed}: 完成")
                        completed_experiments.append({
                            "experiment": exp_name,
                            "dataset": dataset,
                            "seed": seed,
                            "status": "completed",
                            "log_file": str(log_file)
                        })
                    else:
                        self.status["failed"] += 1
                        print(f"   ✗ {exp_name}/{dataset}/seed{seed}: 失败")
                        failed_experiments.append({
                            "experiment": exp_name,
                            "dataset": dataset,
                            "seed": seed,
                            "status": "failed",
                            "log_file": str(log_file)
                        })
                    
                    # 保存进度
                    self.save_progress(completed_experiments, failed_experiments)
        
        # 运行完成
        self.finalize(completed_experiments, failed_experiments)
    
    def save_progress(self, completed: List[Dict], failed: List[Dict]):
        """保存进度到文件"""
        progress = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": self.status.copy(),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "elapsed_time": str(datetime.datetime.now() - self.start_time),
            "completed_experiments": completed[-10:],  # 最近10个
            "failed_experiments": failed[-10:]  # 最近10个失败
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def finalize(self, completed: List[Dict], failed: List[Dict]):
        """实验完成后的处理"""
        elapsed = datetime.datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("COA/BWO 实验运行完成")
        print("="*60)
        
        print(f"\n📊 统计信息:")
        print(f"   总实验数: {self.status['total_experiments']}")
        print(f"   完成: {self.status['completed']}")
        print(f"   失败: {self.status['failed']}")
        print(f"   运行中: {self.status['running']}")
        print(f"   排队中: {self.status['queued']}")
        print(f"   总耗时: {elapsed}")
        
        if failed:
            print(f"\n❌ 失败实验 ({len(failed)}):")
            for f in failed[:5]:  # 只显示前5个
                print(f"   • {f['experiment']}/{f['dataset']}/seed{f['seed']}")
            if len(failed) > 5:
                print(f"   ... 还有 {len(failed)-5} 个失败实验")
        
        # 生成结果汇总
        print(f"\n📁 结果文件:")
        result_files = list(Path(".").glob("*_*.json"))
        if result_files:
            for rf in result_files[:10]:
                size = rf.stat().st_size if rf.exists() else 0
                print(f"   • {rf.name} ({size:,} bytes)")
            if len(result_files) > 10:
                print(f"   ... 还有 {len(result_files)-10} 个结果文件")
        else:
            print("   未找到结果文件")
        
        # 运行结果收集器
        print(f"\n🔄 运行结果收集器...")
        try:
            subprocess.run([sys.executable, "coa_bwo_results_collector.py"], 
                         check=False)
        except Exception as e:
            print(f"   结果收集失败: {e}")
        
        # 保存最终报告
        final_report = {
            "experiment_completed": datetime.datetime.now().isoformat(),
            "elapsed_time": str(elapsed),
            "total_experiments": self.status['total_experiments'],
            "completed": self.status['completed'],
            "failed": self.status['failed'],
            "completed_details": completed,
            "failed_details": failed,
            "log_directory": str(self.log_dir),
            "progress_file": str(self.progress_file)
        }
        
        report_file = self.log_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📄 最终报告已保存: {report_file}")
        print(f"📁 日志目录: {self.log_dir}")
        
        if failed:
            print(f"\n⚠️  有 {len(failed)} 个实验失败，请检查日志文件")
            return 1
        else:
            print(f"\n✅ 所有实验成功完成!")
            return 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行COA/BWO完整实验")
    parser.add_argument("--log-dir", default="experiment_logs",
                       help="日志目录 (默认: experiment_logs)")
    parser.add_argument("--results-dir", default=".",
                       help="结果目录 (默认: 当前目录)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="最大并行实验数 (默认: 1)")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细输出")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已完成的实验")
    parser.add_argument("--background", action="store_true",
                       help="后台运行模式")
    parser.add_argument("--pid-file", default="experiment_runner.pid",
                       help="PID文件路径 (后台模式使用)")
    
    args = parser.parse_args()
    
    # 后台运行处理
    if args.background:
        import daemon
        from daemon.pidfile import TimeoutPIDLockFile
        
        print(f"启动后台实验运行器...")
        print(f"PID文件: {args.pid_file}")
        print(f"日志目录: {args.log_dir}")
        
        # 创建守护进程
        with daemon.DaemonContext(
            working_directory=os.getcwd(),
            umask=0o002,
            pidfile=TimeoutPIDLockFile(args.pid_file),
            stdout=open(os.path.join(args.log_dir, "daemon_stdout.log"), "w+"),
            stderr=open(os.path.join(args.log_dir, "daemon_stderr.log"), "w+")
        ):
            runner = ExperimentRunner(
                log_dir=args.log_dir,
                results_dir=args.results_dir,
                max_parallel=args.parallel
            )
            
            if runner.setup_environment():
                return runner.run_all_experiments(
                    verbose=args.verbose,
                    skip_completed=not args.no_skip
                )
            else:
                return 1
    else:
        # 前台运行
        runner = ExperimentRunner(
            log_dir=args.log_dir,
            results_dir=args.results_dir,
            max_parallel=args.parallel
        )
        
        if runner.setup_environment():
            return runner.run_all_experiments(
                verbose=args.verbose,
                skip_completed=not args.no_skip
            )
        else:
            return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n实验运行被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)