# 2clean_and_merge_v3.py
"""
合并并清洗场景数据 - v3
✅ 支持5密度配置
✅ 流量验证
✅ 【修改1】长期静止处理：删除静止开始之后的所有部分
✅ 【修改2】采集范围调整为60米（与销毁距离一致）

清洗逻辑：
1. 过滤范围外数据 (>60米)
2. 过滤短轨迹 (<2秒)
3. 截断长期静止：检测连续静止≥2秒的起始点，删除该点之后的所有数据
4. 再次过滤短轨迹（截断后可能变短）
5. 重新分配全局trackId
"""
import sys
sys.path.append('D:/Carla Simulation')

import pandas as pd
import numpy as np
from pathlib import Path

# ===== 配置 =====
BASE_DIR = 'D:/Carla Simulation'
RAW_DATA_DIR = f'{BASE_DIR}/data/raw_v5_simultaneous'
PROCESSED_DATA_DIR = f'{BASE_DIR}/data/processed_v5_simultaneous'

COLLECTION_RADIUS = 60.0  # 【修改】采集范围改为60米
FRAME_RATE = 10

# 清洗参数
MIN_TRACK_LENGTH = 20          # 最小轨迹长度（帧）= 2秒
STATIC_SPEED_THRESHOLD = 0.5   # 静止速度阈值 (m/s)
STATIC_DURATION_THRESHOLD = 20 # 长期静止阈值（帧）= 2秒

# 密度配置（用于验证）
TRAFFIC_DENSITIES = {
    'very_sparse': {'target_flow': 300, 'target_passages': 15},
    'sparse': {'target_flow': 500, 'target_passages': 25},
    'medium': {'target_flow': 750, 'target_passages': 38},
    'dense': {'target_flow': 1050, 'target_passages': 53},
    'very_dense': {'target_flow': 1400, 'target_passages': 70},
}

TOTAL_SCENARIOS = 25  # 5天气 × 5密度


def load_all_scenarios():
    """加载所有场景数据"""
    print(f"正在加载场景数据...")
    print(f"数据目录: {RAW_DATA_DIR}")

    all_data = []
    for i in range(TOTAL_SCENARIOS):
        file_path = Path(RAW_DATA_DIR) / f'scenario_{i:03d}.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['scenario_id'] = i
            all_data.append(df)
            print(f"  ✓ scenario_{i:03d}.csv: {len(df):,} 行, {df['trackId'].nunique()} 轨迹")
        else:
            print(f"  ✗ scenario_{i:03d}.csv: 未找到")

    if not all_data:
        print("❌ 没有找到任何数据文件")
        return None

    merged = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ 加载完成: {len(merged):,} 行, {merged['trackId'].nunique()} 条轨迹")

    return merged


def truncate_at_long_static(track_df, speed_threshold=0.5, duration_threshold=20):
    """
    截断长期静止的轨迹
    
    逻辑：
    1. 找到连续静止（速度<阈值）超过duration_threshold帧的起始点
    2. 删除该点之后的【所有】数据（包括之后又行驶的部分）
    3. 只保留静止之前的有效轨迹
    
    示例：
        帧1-100:   行驶 5m/s           → ✅ 保留
        帧101-130: 静止 0m/s (≥2秒)    → ❌ 删除
        帧131-200: 又行驶 5m/s         → ❌ 删除（静止之后全删）
    
    参数：
        track_df: 单条轨迹的DataFrame
        speed_threshold: 静止速度阈值 (m/s)
        duration_threshold: 连续静止帧数阈值
    
    返回：
        截断后的DataFrame（可能为空）
    """
    if len(track_df) == 0:
        return track_df
    
    # 按帧排序
    track_df = track_df.sort_values('frame').reset_index(drop=True)
    
    # 标记静止帧
    is_static = track_df['speed'] < speed_threshold
    
    # 找连续静止段
    static_start = None
    consecutive_static = 0
    truncate_index = None
    
    for i, static in enumerate(is_static):
        if static:
            if static_start is None:
                static_start = i
            consecutive_static += 1
            
            # 检测到长期静止，记录截断点
            if consecutive_static >= duration_threshold:
                truncate_index = static_start
                break
        else:
            # 重置计数
            static_start = None
            consecutive_static = 0
    
    # 截断：只保留静止开始之前的数据
    if truncate_index is not None:
        track_df = track_df.iloc[:truncate_index]
    
    return track_df


def clean_data(df):
    """清洗数据"""
    print("\n" + "=" * 80)
    print("数据清洗")
    print("=" * 80)

    original_rows = len(df)
    original_tracks = df['trackId'].nunique()

    print(f"\n原始数据: {original_rows:,} 行, {original_tracks} 条轨迹")

    # ═══════════════════════════════════════════════════════════════════
    # [1/5] 过滤范围外数据
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[1/5] 过滤范围外数据 (>{COLLECTION_RADIUS}米)...")
    df_filtered = df[df['radius'] <= COLLECTION_RADIUS].copy()
    removed = original_rows - len(df_filtered)
    print(f"  移除 {removed:,} 行 ({removed / original_rows * 100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # [2/5] 过滤短轨迹（第一次）
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[2/5] 过滤短轨迹 (<{MIN_TRACK_LENGTH}帧 = {MIN_TRACK_LENGTH/FRAME_RATE:.1f}秒)...")
    track_lengths = df_filtered.groupby('trackId').size()
    valid_tracks = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
    before_tracks = df_filtered['trackId'].nunique()
    df_filtered = df_filtered[df_filtered['trackId'].isin(valid_tracks)]
    removed_tracks = before_tracks - df_filtered['trackId'].nunique()
    print(f"  移除 {removed_tracks} 条短轨迹")

    # ═══════════════════════════════════════════════════════════════════
    # [3/5] 截断长期静止
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[3/5] 截断长期静止 (连续≥{STATIC_DURATION_THRESHOLD}帧 速度<{STATIC_SPEED_THRESHOLD}m/s)...")
    print(f"  【注意】静止之后的所有数据都会被删除（包括之后又行驶的部分）")
    print(f"  处理中...")
    
    before_rows = len(df_filtered)
    before_tracks = df_filtered['trackId'].nunique()
    
    # 按轨迹分组处理
    truncated_dfs = []
    truncated_count = 0
    
    for track_id, track_df in df_filtered.groupby('trackId'):
        original_len = len(track_df)
        truncated_df = truncate_at_long_static(
            track_df, 
            speed_threshold=STATIC_SPEED_THRESHOLD,
            duration_threshold=STATIC_DURATION_THRESHOLD
        )
        
        if len(truncated_df) < original_len:
            truncated_count += 1
        
        if len(truncated_df) > 0:
            truncated_dfs.append(truncated_df)
    
    if truncated_dfs:
        df_filtered = pd.concat(truncated_dfs, ignore_index=True)
    else:
        df_filtered = pd.DataFrame()
    
    removed_rows = before_rows - len(df_filtered)
    print(f"  截断了 {truncated_count} 条轨迹")
    print(f"  移除 {removed_rows:,} 行数据 ({removed_rows / before_rows * 100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # [4/5] 再次过滤短轨迹（截断后可能变短）
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[4/5] 再次过滤短轨迹 (<{MIN_TRACK_LENGTH}帧)...")
    if len(df_filtered) > 0:
        track_lengths = df_filtered.groupby('trackId').size()
        valid_tracks = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
        before_tracks = df_filtered['trackId'].nunique()
        df_filtered = df_filtered[df_filtered['trackId'].isin(valid_tracks)]
        removed_tracks = before_tracks - df_filtered['trackId'].nunique()
        print(f"  移除 {removed_tracks} 条变短的轨迹")
    else:
        print(f"  无数据需要处理")

    # ═══════════════════════════════════════════════════════════════════
    # [5/5] 重新分配全局trackId
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[5/5] 重新分配全局trackId...")
    if len(df_filtered) > 0:
        df_filtered['original_trackId'] = df_filtered['trackId']
        df_filtered['global_trackId'] = (
            df_filtered['scenario_id'].astype(str) + '_' +
            df_filtered['trackId'].astype(str)
        )
        track_mapping = {track: idx for idx, track in enumerate(df_filtered['global_trackId'].unique())}
        df_filtered['trackId'] = df_filtered['global_trackId'].map(track_mapping)
        df_filtered = df_filtered.drop(columns=['global_trackId'])
        print(f"  分配了 {len(track_mapping)} 个全局ID")

    # 统计结果
    final_rows = len(df_filtered)
    final_tracks = df_filtered['trackId'].nunique() if len(df_filtered) > 0 else 0

    print("\n" + "=" * 80)
    print("清洗结果")
    print("=" * 80)
    print(f"\n数据行数: {original_rows:,} → {final_rows:,} (保留{final_rows/original_rows*100:.1f}%)")
    print(f"轨迹数量: {original_tracks} → {final_tracks} (保留{final_tracks/original_tracks*100:.1f}%)")

    return df_filtered


def verify_flow_rates(df):
    """验证流量是否符合目标"""
    print("\n" + "=" * 80)
    print("流量验证 (基于HCM 2010目标)")
    print("=" * 80)
    
    print(f"\n核心区定义: 半径 ≤ 25米")
    
    results = []
    
    for scenario_id in sorted(df['scenario_id'].unique()):
        scenario_data = df[df['scenario_id'] == scenario_id]
        
        density = scenario_data['traffic_density'].iloc[0]
        weather = scenario_data['weather'].iloc[0]
        
        # 统计核心区轨迹
        core_data = scenario_data[scenario_data['radius'] <= 25]
        core_tracks = core_data['trackId'].nunique()
        
        # 获取目标值
        target = TRAFFIC_DENSITIES[density]['target_passages']
        
        # 计算偏差
        deviation = (core_tracks - target) / target * 100 if target > 0 else 0
        status = "✅" if abs(deviation) <= 30 else "⚠️"
        
        results.append({
            'scenario_id': scenario_id,
            'weather': weather,
            'density': density,
            'target': target,
            'actual': core_tracks,
            'deviation': deviation,
            'status': status
        })
    
    results_df = pd.DataFrame(results)
    
    # 按密度统计
    print("\n按密度统计:")
    print(f"  {'密度':<12} {'目标':<6} {'平均实际':<10} {'达成率':<10}")
    print(f"  {'-' * 45}")
    
    for density in TRAFFIC_DENSITIES.keys():
        density_results = results_df[results_df['density'] == density]
        if len(density_results) == 0:
            continue
            
        target = TRAFFIC_DENSITIES[density]['target_passages']
        avg_actual = density_results['actual'].mean()
        rate = avg_actual / target * 100 if target > 0 else 0
        
        print(f"  {density:<12} {target:<6} {avg_actual:<10.1f} {rate:<10.1f}%")
    
    # 保存报告
    report_file = Path(PROCESSED_DATA_DIR) / 'flow_validation_report.csv'
    results_df.to_csv(report_file, index=False)
    print(f"\n✅ 验证报告已保存: {report_file}")
    
    return results_df


def analyze_data(df):
    """分析清洗后的数据"""
    print("\n" + "=" * 80)
    print("数据分析")
    print("=" * 80)

    print(f"\n📊 基本统计:")
    print(f"  总行数: {len(df):,}")
    print(f"  轨迹数: {df['trackId'].nunique()}")
    print(f"  场景数: {df['scenario_id'].nunique()}")

    print(f"\n📈 运动统计:")
    print(f"  速度: {df['speed'].mean():.2f} m/s ({df['speed'].mean()*3.6:.1f} km/h)")
    print(f"  速度范围: {df['speed'].min():.2f} - {df['speed'].max():.2f} m/s")
    print(f"  平均半径: {df['radius'].mean():.2f} m")

    print(f"\n🌦️ 天气分布:")
    for weather in sorted(df['weather'].unique()):
        count = len(df[df['weather'] == weather])
        tracks = df[df['weather'] == weather]['trackId'].nunique()
        print(f"  {weather:20s}: {count:7,} 行, {tracks:4} 轨迹")

    print(f"\n🚗 密度分布:")
    for density in TRAFFIC_DENSITIES.keys():
        if density in df['traffic_density'].values:
            count = len(df[df['traffic_density'] == density])
            tracks = df[df['traffic_density'] == density]['trackId'].nunique()
            print(f"  {density:12s}: {count:7,} 行, {tracks:4} 轨迹")

    print(f"\n🎯 行为分布:")
    for behavior in sorted(df['behavior_type'].unique()):
        count = len(df[df['behavior_type'] == behavior])
        tracks = df[df['behavior_type'] == behavior]['trackId'].nunique()
        avg_speed = df[df['behavior_type'] == behavior]['speed'].mean()
        print(f"  {behavior:10s}: {count:7,} 行, {tracks:4} 轨迹, {avg_speed:.2f} m/s")
    
    # 核心区统计
    print(f"\n🎯 核心区统计 (≤25米):")
    core_data = df[df['radius'] <= 25]
    core_tracks = core_data['trackId'].nunique()
    print(f"  核心区数据: {len(core_data):,} 行")
    print(f"  进入核心区轨迹: {core_tracks} 条")
    print(f"  核心区占比: {len(core_data)/len(df)*100:.1f}%")


def main():
    print("=" * 80)
    print("数据合并、清洗与验证 - v3")
    print("=" * 80)
    
    print(f"\n📋 清洗参数:")
    print(f"  采集范围: ≤{COLLECTION_RADIUS}米")
    print(f"  最小轨迹长度: {MIN_TRACK_LENGTH}帧 ({MIN_TRACK_LENGTH/FRAME_RATE:.1f}秒)")
    print(f"  静止速度阈值: {STATIC_SPEED_THRESHOLD} m/s")
    print(f"  长期静止阈值: {STATIC_DURATION_THRESHOLD}帧 ({STATIC_DURATION_THRESHOLD/FRAME_RATE:.1f}秒)")
    print(f"  【注意】长期静止之后的所有数据都会被删除")

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    df_raw = load_all_scenarios()
    if df_raw is None:
        return

    # 2. 清洗数据
    df_clean = clean_data(df_raw)
    
    if len(df_clean) == 0:
        print("❌ 清洗后无数据")
        return

    # 3. 验证流量
    verify_flow_rates(df_clean)

    # 4. 分析数据
    analyze_data(df_clean)

    # 5. 保存
    output_file = Path(PROCESSED_DATA_DIR) / 'carla_round_all.csv'
    df_clean.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("✅ 保存完成")
    print("=" * 80)
    print(f"\n文件: {output_file}")
    print(f"大小: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n下一步: 运行 3split_dataset.py 划分数据集")


if __name__ == '__main__':
    main()
